<!--
 * @Author: HU Zheng
 * @Date: 2023-01-15 13:51:47
 * @LastEditors: HU Zheng
 * @LastEditTime: 2023-01-15 13:53:55
 * @Description: file content
-->
# 最优化PJ

罗子钦 22210980110

胡正 22210980105

## 仓库结构

-   Unused PJ过程中的中间文件

-   Datasets 数据集

-   Raw_results 实验生成的生结果

-   results_analysis 分析绘图用的表格

## 代码使用示例

非随机方法实验：

```shell
python runall_non_stoch.py \--data_path datasets/w8a \--save_path
D_100_w8a.json \--diameter 100 \--lamda 100 \--rm_zeros 0
```
随机试验：

```shell
python run_stoch.py \--data_path datasets/covtype.libsvm.binary.scale
\--save_path cov_D\_500.json \--diameter 500 \--lamda 100 \--rm_zeros 0
```
gisette实验：

```shell
python runipm.py \--data_path datasets/gisette_scale \--save_path
D_10_gissette.json \--diameter 10 \--lamda 100 \--rm_zeros 0
```
w8a.t实验：

```shell
python runctr_ipm.py \--data_path datasets/w8a.t \--save_path
D_100_w8at.json \--diameter 100 \--lamda 100 \--rm_zeros 0
```
