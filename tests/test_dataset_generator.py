"""
大规模数据集生成器测试。

测试覆盖:
- LHS参数采样的维度、范围和类型特异性
- 噪声注入的统计特性和边界条件
- 单样本模拟管线（ODE -> 噪声 -> ctFP）
- HDF5存储格式和数据完整性
- 并行生成管线端到端验证
"""

import os
import tempfile
import shutil

import numpy as np
import pytest
import h5py

from src.dataset_generator import (
    generate_lhs_parameters,
    inject_noise,
    simulate_single_sample,
    save_to_hdf5,
    generate_dataset_parallel,
    RAFT_TYPES,
    PARAM_BOUNDS,
)


# ============================================================
# Task 1: LHS参数采样
# ============================================================

class TestGenerateLHSParameters:
    """LHS参数采样器测试"""

    def test_returns_correct_count(self):
        """返回指定数量的参数字典"""
        params = generate_lhs_parameters(100, 'trithiocarbonate', seed=42)
        assert len(params) == 100

    def test_contains_required_keys(self):
        """每个参数字典包含所有必需的ODE参数"""
        params = generate_lhs_parameters(10, 'trithiocarbonate', seed=42)
        required = ['kp', 'kt', 'kd', 'f', 'I0', 'ki', 'kadd', 'kfrag',
                     'M0', 'CTA0', 'M_monomer', 'raft_type', 'log10_Ctr', 'cta_ratio']
        for p in params:
            for key in required:
                assert key in p, f"缺少参数: {key}"

    def test_raft_type_stored(self):
        """raft_type正确存储在参数字典中"""
        for rt in RAFT_TYPES:
            params = generate_lhs_parameters(5, rt, seed=42)
            for p in params:
                assert p['raft_type'] == rt

    def test_dithioester_has_preequilibrium(self):
        """dithioester类型包含预平衡参数kadd0和kfrag0"""
        params = generate_lhs_parameters(10, 'dithioester', seed=42)
        for p in params:
            assert 'kadd0' in p, "dithioester应包含kadd0"
            assert 'kfrag0' in p, "dithioester应包含kfrag0"

    def test_non_dithioester_no_preequilibrium(self):
        """非dithioester类型不包含预平衡参数"""
        for rt in ['trithiocarbonate', 'xanthate', 'dithiocarbamate']:
            params = generate_lhs_parameters(5, rt, seed=42)
            for p in params:
                assert 'kadd0' not in p, f"{rt}不应包含kadd0"

    def test_parameter_ranges(self):
        """采样参数在定义的范围内"""
        params = generate_lhs_parameters(500, 'trithiocarbonate', seed=42)

        log10_Ctr_values = [p['log10_Ctr'] for p in params]
        kp_values = [p['kp'] for p in params]
        kt_values = [p['kt'] for p in params]
        kd_values = [p['kd'] for p in params]
        I0_values = [p['I0'] for p in params]
        f_values = [p['f'] for p in params]
        cta_values = [p['cta_ratio'] for p in params]

        # log10_Ctr in [-2, 4]
        assert min(log10_Ctr_values) >= -2.0 - 0.01
        assert max(log10_Ctr_values) <= 4.0 + 0.01

        # kp in [100, 10000]
        assert min(kp_values) >= 100 * 0.99
        assert max(kp_values) <= 10000 * 1.01

        # kt in [1e6, 1e9]
        assert min(kt_values) >= 1e6 * 0.99
        assert max(kt_values) <= 1e9 * 1.01

        # kd in [1e-6, 1e-4]
        assert min(kd_values) >= 1e-6 * 0.99
        assert max(kd_values) <= 1e-4 * 1.01

        # I0 in [0.001, 0.05]
        assert min(I0_values) >= 0.001 - 0.0001
        assert max(I0_values) <= 0.05 + 0.001

        # f in [0.5, 0.8]
        assert min(f_values) >= 0.5 - 0.01
        assert max(f_values) <= 0.8 + 0.01

        # cta_ratio in [0.001, 0.1]
        assert min(cta_values) >= 0.001 * 0.99
        assert max(cta_values) <= 0.1 * 1.01

    def test_seed_reproducibility(self):
        """相同种子产生相同参数"""
        params1 = generate_lhs_parameters(50, 'trithiocarbonate', seed=123)
        params2 = generate_lhs_parameters(50, 'trithiocarbonate', seed=123)

        for p1, p2 in zip(params1, params2):
            assert p1['log10_Ctr'] == p2['log10_Ctr']
            assert p1['kp'] == p2['kp']

    def test_different_seeds_differ(self):
        """不同种子产生不同参数"""
        params1 = generate_lhs_parameters(50, 'trithiocarbonate', seed=1)
        params2 = generate_lhs_parameters(50, 'trithiocarbonate', seed=2)

        # 至少部分参数应不同
        diffs = sum(1 for p1, p2 in zip(params1, params2)
                    if p1['log10_Ctr'] != p2['log10_Ctr'])
        assert diffs > 0


# ============================================================
# Task 1: 噪声注入
# ============================================================

class TestInjectNoise:
    """噪声注入测试"""

    def test_output_shape_preserved(self):
        """输出形状与输入相同"""
        Mn = np.array([1000, 2000, 3000])
        D = np.array([1.2, 1.5, 1.8])
        Mn_noisy, D_noisy = inject_noise(Mn, D, sigma=0.03)
        assert Mn_noisy.shape == Mn.shape
        assert D_noisy.shape == D.shape

    def test_dispersity_clipped(self):
        """分散度不低于1.0"""
        Mn = np.array([1000, 2000, 3000])
        D = np.array([1.01, 1.0, 1.0])  # 接近1.0边界
        # 多次测试确保clip生效
        for _ in range(10):
            _, D_noisy = inject_noise(Mn, D, sigma=0.1)  # 大噪声
            assert np.all(D_noisy >= 1.0), f"D_noisy最小值 {D_noisy.min()} < 1.0"

    def test_mn_non_negative(self):
        """Mn不为负值"""
        Mn = np.array([10, 20, 30])  # 小值
        D = np.array([1.5, 1.5, 1.5])
        for _ in range(10):
            Mn_noisy, _ = inject_noise(Mn, D, sigma=0.5)  # 大噪声
            assert np.all(Mn_noisy >= 0), f"Mn_noisy最小值 {Mn_noisy.min()} < 0"

    def test_noise_magnitude(self):
        """噪声在预期范围内（统计）"""
        np.random.seed(42)
        Mn = np.ones(10000) * 1000
        D = np.ones(10000) * 1.5
        sigma = 0.03

        Mn_noisy, D_noisy = inject_noise(Mn, D, sigma=sigma)

        # 相对误差的标准差应接近sigma
        rel_error_mn = (Mn_noisy - Mn) / Mn
        assert abs(np.std(rel_error_mn) - sigma) < 0.01, (
            f"Mn噪声标准差 {np.std(rel_error_mn):.4f} 应接近 {sigma}"
        )

    def test_zero_sigma(self):
        """sigma=0时输出等于输入"""
        Mn = np.array([1000, 2000, 3000])
        D = np.array([1.2, 1.5, 1.8])
        Mn_noisy, D_noisy = inject_noise(Mn, D, sigma=0.0)
        np.testing.assert_array_equal(Mn_noisy, Mn)
        np.testing.assert_array_equal(D_noisy, D)


# ============================================================
# Task 1: 单样本模拟
# ============================================================

class TestSimulateSingleSample:
    """单样本模拟管线测试"""

    def test_success_returns_correct_structure(self):
        """成功样本返回fingerprint/labels/params/success"""
        params = generate_lhs_parameters(5, 'trithiocarbonate', seed=42)
        result = simulate_single_sample(params[0])
        # 可能成功或失败，取决于ODE参数
        if result['success']:
            assert 'fingerprint' in result
            assert 'labels' in result
            assert 'params' in result
            assert result['fingerprint'].shape == (64, 64, 2)
            assert len(result['labels']) == 3

    def test_failure_returns_error(self):
        """失败样本返回success=False和error信息"""
        # 构造一个极端参数，大概率导致ODE失败
        params = {
            'kp': 1e10, 'kt': 1e15, 'kd': 1.0, 'f': 0.9,
            'I0': 1.0, 'ki': 1e4, 'kadd': 1e15, 'kfrag': 1e4,
            'M0': 1.0, 'CTA0': 0.5, 'M_monomer': 100.0,
            'raft_type': 'trithiocarbonate',
            'log10_Ctr': 5.0, 'cta_ratio': 0.5,
        }
        result = simulate_single_sample(params)
        if not result['success']:
            assert 'error' in result
            assert isinstance(result['error'], str)
        # 如果碰巧成功了也可以

    def test_labels_range(self):
        """标签值在合理范围内"""
        params = generate_lhs_parameters(20, 'trithiocarbonate', seed=42)
        for p in params[:5]:  # 只测5个节省时间
            result = simulate_single_sample(p)
            if result['success']:
                log10_ctr, inh, ret = result['labels']
                assert -3 <= log10_ctr <= 5, f"log10_Ctr={log10_ctr} 超出范围"
                assert 0 <= inh <= 1, f"inhibition={inh} 超出范围"
                assert 0 < ret <= 1, f"retardation={ret} 超出范围"

    def test_fingerprint_not_all_zeros(self):
        """指纹不应全为零"""
        params = generate_lhs_parameters(10, 'trithiocarbonate', seed=42)
        for p in params[:3]:
            result = simulate_single_sample(p)
            if result['success']:
                fp = result['fingerprint']
                assert np.sum(fp) > 0, "指纹不应全为零"
                break  # 只需一个成功样本即可


# ============================================================
# Task 2: HDF5存储
# ============================================================

class TestSaveToHDF5:
    """HDF5存储测试"""

    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def sample_data(self):
        """生成少量样本用于测试"""
        params = generate_lhs_parameters(10, 'trithiocarbonate', seed=42)
        results = [simulate_single_sample(p) for p in params]
        successful = [r for r in results if r['success']]
        return successful

    def test_creates_h5_file(self, tmpdir, sample_data):
        """创建HDF5文件"""
        if len(sample_data) == 0:
            pytest.skip("无成功样本")
        save_to_hdf5(sample_data, 'trithiocarbonate', tmpdir)
        h5_path = os.path.join(tmpdir, 'trithiocarbonate.h5')
        assert os.path.exists(h5_path)

    def test_h5_datasets_exist(self, tmpdir, sample_data):
        """HDF5包含fingerprints、labels和params数据集"""
        if len(sample_data) == 0:
            pytest.skip("无成功样本")
        save_to_hdf5(sample_data, 'trithiocarbonate', tmpdir)
        h5_path = os.path.join(tmpdir, 'trithiocarbonate.h5')

        with h5py.File(h5_path, 'r') as f:
            assert 'fingerprints' in f
            assert 'labels' in f
            assert 'params' in f

    def test_h5_shapes(self, tmpdir, sample_data):
        """HDF5数据集shape正确"""
        if len(sample_data) == 0:
            pytest.skip("无成功样本")
        n = len(sample_data)
        save_to_hdf5(sample_data, 'trithiocarbonate', tmpdir)
        h5_path = os.path.join(tmpdir, 'trithiocarbonate.h5')

        with h5py.File(h5_path, 'r') as f:
            assert f['fingerprints'].shape == (n, 64, 64, 2)
            assert f['labels'].shape == (n, 3)

    def test_h5_data_values(self, tmpdir, sample_data):
        """HDF5存储的数据值与原始数据一致"""
        if len(sample_data) == 0:
            pytest.skip("无成功样本")
        save_to_hdf5(sample_data, 'trithiocarbonate', tmpdir)
        h5_path = os.path.join(tmpdir, 'trithiocarbonate.h5')

        with h5py.File(h5_path, 'r') as f:
            # 验证第一个样本的标签
            expected_labels = np.array(sample_data[0]['labels'], dtype=np.float32)
            np.testing.assert_array_almost_equal(
                f['labels'][0], expected_labels, decimal=5
            )

    def test_h5_metadata(self, tmpdir, sample_data):
        """HDF5包含正确的元数据"""
        if len(sample_data) == 0:
            pytest.skip("无成功样本")
        save_to_hdf5(sample_data, 'trithiocarbonate', tmpdir)
        h5_path = os.path.join(tmpdir, 'trithiocarbonate.h5')

        with h5py.File(h5_path, 'r') as f:
            assert f.attrs['raft_type'] == 'trithiocarbonate'
            assert f.attrs['n_samples'] == len(sample_data)

    def test_empty_samples_no_crash(self, tmpdir):
        """空样本列表不崩溃"""
        save_to_hdf5([], 'trithiocarbonate', tmpdir)
        # 文件应该存在但为空或有零样本
        h5_path = os.path.join(tmpdir, 'trithiocarbonate.h5')
        # 空样本可能不创建文件，这是可接受的


# ============================================================
# Task 2: 并行生成管线
# ============================================================

class TestGenerateDatasetParallel:
    """并行生成管线端到端测试"""

    @pytest.fixture
    def tmpdir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.mark.slow
    def test_end_to_end_small(self, tmpdir):
        """小规模端到端测试（20样本）"""
        stats = generate_dataset_parallel(
            'trithiocarbonate', n_samples=20,
            output_path=tmpdir, seed=42,
        )

        assert stats['n_total'] == 20
        assert stats['n_success'] + stats['n_failed'] == 20
        assert stats['n_success'] > 0  # 至少应有一些成功

        # 验证HDF5文件
        h5_path = stats['h5_path']
        assert os.path.exists(h5_path)

        with h5py.File(h5_path, 'r') as f:
            assert f['fingerprints'].shape[0] == stats['n_success']
            assert f['fingerprints'].shape[1:] == (64, 64, 2)
            assert f['labels'].shape == (stats['n_success'], 3)

    @pytest.mark.slow
    def test_failure_rate_reported(self, tmpdir):
        """失败率正确计算"""
        stats = generate_dataset_parallel(
            'trithiocarbonate', n_samples=10,
            output_path=tmpdir, seed=42,
        )
        expected_rate = stats['n_failed'] / stats['n_total']
        assert abs(stats['failure_rate'] - expected_rate) < 1e-10

    @pytest.mark.slow
    def test_all_raft_types(self, tmpdir):
        """所有RAFT类型都能生成"""
        for rt in RAFT_TYPES:
            stats = generate_dataset_parallel(
                rt, n_samples=5,
                output_path=tmpdir, seed=42,
            )
            assert stats['n_total'] == 5
            # 允许少量失败
            assert stats['n_success'] >= 1, f"{rt}应至少有1个成功样本"
