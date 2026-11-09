# qldpc_generator

用于复现 qLDPC 综述中的有限构造、矩阵参数和构造之间的基变换。主要参考版本为 **2026-11-07 clean.tex**，版本摘要保存在 `paper_cases.PAPER_SOURCE`。

采用三层结构：**库函数负责构造和计算，自动测试负责核验，notebook 负责讲解与展示。** Notebook 调用库函数，不复制一份构造算法。

## 安装与运行

Python 3.10 或更新版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,notebooks]'
python -m pytest -q
python -m qldpc_generator.verify_paper
python scripts/execute_notebooks.py
```

最后一条命令使用独立 Jupyter 内核运行每个 notebook，并保存执行结果。若已安装 JupyterLab，也可以直接打开 `notebooks/`。单独安装此项目不会自动安装 JupyterLab。

验证报告写入 `reports/paper_validation.json`。使用 `--output PATH` 指定位置，或使用 `--skip-distance` 只核验代数和对应关系。报告包含软件版本、源代码哈希、论文版本、每项结果及精确距离的逻辑算子证书。

`requirements-tested.txt` 记录本轮 macOS / Python 3.14 的实际依赖版本，供追溯使用；跨平台安装以 `pyproject.toml` 为准。

```python
import numpy as np
from qldpc_generator import hgp

H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
code = hgp(H, H)
print(code.parameters())       # N=13, K=1, 两个秩均为 6
print(code.distance(3))        # exact, D=3，含 X/Z 逻辑算子支撑
LX, LZ = code.logical_operators()
assert np.array_equal(LX @ LZ.T % 2, np.eye(code.k, dtype=int))
```

## 当前提供的内容

| 内容 | 库接口 / 模块 | 验证重点 |
| --- | --- | --- |
| GF(2)、CSS 和子系统码基础 | `gf2`, `CSSCode`, `subsystem_parameters` | 二元秩、核、稳定子商空间、逻辑配对、规范与稳定子维数 |
| 循环矩阵、群代数、Tanner 图 | `algebra`, `representations` | 移位方向、负指数、模 2 抵消、左右乘法、标号图往返 |
| HGP | `hgp`, `hgp_distance_witness` | 矩形/秩亏输入、转置码维数、距离证明中的商映射 |
| LP / LCS | `lifted_product`, `group_lifted_product`, `lcs` | 形式伴随、共享提升坐标、S3 非交换例子、三个大 LP 实例 |
| FB | `fiber_bundle` | 每个非零基关联上的 twist、两个纤维链空间的独立作用、链自同构条件 |
| BP | `balanced_product` | 单源代表元、模 2 边界、偶数阶群、非自由作用、代表元独立性 |
| 构造对应 | `lp_to_bp`, `bp_to_lp`, `fb_to_bp`, `bp_to_fb` | 显式 P0/P1/P2；同一个 P1 同时重排 HX 和 HZ；反向构造检查假设 |
| Bicycle / GB / BB | `standard_bicycle`, `generalized_bicycle`, `bivariate_bicycle` | 多项式 gcd、独立 Gröbner 商维数、反足映射、直和分解 |
| 相关变体 | `generalized_toric`, `toric`, `coprime_bicycle`, `planar_surface` | 单项式平移等价、互素 CRT 重排、周期/开放重复链边界 |
| 论文实例 | `paper_cases`, `verify_paper` | 12 个明确给出输入的实例；逐项区分距离结论状态 |

实现使用稠密二元矩阵，面向论文中的有限实例和可审计验证；不是超大规模稀疏码生成器或解码性能基准。

## 数学约定与接口

- `code.hx` 和 `code.hz` 都是检查为行、量子比特为列；构造时验证 `HX @ HZ.T % 2 == 0`。矩阵输入须为 0/1，所有秩都在 GF(2) 上计算。
- HGP 的输入是 `H1, H2`，都采用经典检查矩阵方向；量子比特按 CC、VV 排序。
- 循环移位满足 `S e_j = e_(j+1 mod l)`。多项式系数模 2，指数按周期约化；支持整数 Laurent 负指数。
- LP 的 A、B 是环上输入，先在共享环上构造，再二元展开。`formal_adjoint` 同时做矩阵转置和 `x -> x^-1`。
- 群代数输入系数张量形状为 `(rows, columns, group_order)`，或用 `group_algebra_matrix(group, entries)` 从“每格群元素索引列表”构造。重复项抵消。默认 A 右乘、B 左乘；mitten 实例显式交换两侧。
- 群作用数组形状为 `(group_order, basis_size)`，保存旧基索引到新基索引。BP 的 `right=(C0,C1)`、`left=(D0,D1)` 可有不同维数，均验证作用律及边界等变性。
- FB 的 `twists[a,b]` 是关联 `b -> a` 的群索引。非关联处必须为 -1；省略 twists 表示全为单位元。
- 构造对应返回的置换向量均为旧索引到新索引。`Equivalence.verify()` 比较两套完整矩阵，不以 N/K 相等代替对应证明。

### 旧代码迁移

旧 `HGPGenerator`、`BPGenerator` 的 `get_parity_matrices()` 仍保留返回顺序 `(旧 HxT, 旧 Hz)`。它实际对应论文的 **`(HZ.T, HX)`**。新代码请使用函数接口，或 `generator.get_css_code().hx/.hz`。`show_parity_matrices()` 使用论文方向。

旧循环矩阵函数的移位方向与论文相反，已改为论文的 `S e_j = e_(j+1)`。需要保持旧数组方向时，对新矩阵取转置；依赖旧索引顺序的保存数据应重新生成。旧 BP 的错误轨道合并算法也已替换，结果可能与旧错误输出不同。

`BPGenerator` 兼容入口只支持同一度空间可共用作用的方阵情形；一般情况使用 `balanced_product()`。历史 k1/k2 仅保留为输入元数据，实际参数由矩阵重新计算。

原 `FBGenerator` 是不可运行的随机 twist 草稿，现使用显式、可复现的新签名：`FBGenerator(base, fiber, group, action0, action1, twists=None)`。不会再随机生成未检查的连接。

## 距离结果怎样阅读

`CSSCode.distance(max_weight, max_subsets)` 是穷尽的分半碰撞搜索。相同 syndrome、不同稳定子商空间签名给出非平凡逻辑算子；对所有更小支撑的排除给出最小性证书。测试另用小码的全空间枚举交叉检查此算法。

- `exact`：在给定实例上排除了更小重量，并给出相应逻辑算子。不是一般族的证明。
- `bounded`：只得到下界，或两扇区结合后的上下界；不能读作精确距离。
- `no_logicals`：K=0，不定义有限非平凡逻辑距离。
- `not_searched`：本次没有执行距离搜索。
- `reported` / `reported_upper_bound`：论文引用的文献值，单独保存在 `distance_claim` 中，不能当成这次程序认证的结果。
- LCS 的一般式在构造元数据中保留 `conjectured`。

默认报告的精确距离为 HGP13 的 3、LCS52 的 4、bicycle6 的 2、GB48 的 8、BB72 的 6。BP90、BP108、BB144 和 mitten150 默认排除重量不超过 6 的逻辑算子，给出 D≥7。三个大 LP 实例只独立核验矩阵、秩和 K，所引距离上界尚无本地证书。

搜索在组合数量达到预算前停止，并报告 `checked_through` 和停止原因。提高重量限制可能带来指数级开销；未找到逻辑算子不等于没有逻辑算子。

## 示例入口

1. [HGP 与距离证明](notebooks/01-hgp-and-distance.ipynb)
2. [LP、LCS 与提升顺序](notebooks/02-lp-lcs.ipynb)
3. [BP 轨道与有限实例](notebooks/03-example-bp.ipynb)
4. [FB、LP–BP、FB–BP 对应](notebooks/04-fb-and-correspondences.ipynb)
5. [Bicycle、BB、商维数和直和](notebooks/05-bicycle-bb.ipynb)
6. [论文验证总表](notebooks/06-paper-validation.ipynb)

## 验证边界与待补输入

这里“构造齐全”指五个核心构造及上述变体、它们的有限矩阵验证和对应条件，不表示论文每条文献结论都已独立证明。

- GB `[[126,28,8]]` 在本稿只列参数，缺少明确的两个多项式。
- 开放边界 `[[88,6,6]]` 与 tile `[[288,8,12]]`、`[[512,18,19]]` 缺少完整边界检查数据。`planar_surface` 是重复链 HGP，不是这几个实例的替代实现。取得明确 HX/HZ 后可交给 `CSSCode` 及距离验证器检查。
- 渐近率/距离定理、一般 LCS 距离猜想、解码阈值、单轮纠错和硬件实验属于另外的证明、仿真或数据复现任务。

详见 [实施路线与仓库审计](IMPLEMENTATION_PLAN.md)。GitHub Actions 工作流已经配置；本地通过不代表远程 CI 已运行。
