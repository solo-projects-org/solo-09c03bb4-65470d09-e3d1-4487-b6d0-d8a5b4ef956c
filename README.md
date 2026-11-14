# 周期变星候选筛选(Lomb–Scargle + 食双星形态判别)

面向天文巡测数据员的网页工具,包含两条互补流水线:

1. **周期筛查**:上传单颗星的测光序列,自动搜索周期信号,输出最高峰周期、
   功率谱、相位折叠散点与置换尾概率;
2. **食双星形态判别**:针对周期筛查相位图中的两个疑似食谷,上传同一颗星含
   `night_id` 的多夜测光,分别按 P 与 2P 折相,做同夜基线校正后的交替食谷
   深度比较并结合两种折相的拟合表现,判断是否值得按食双星复核。

## 快速开始

```bash
docker compose up --build
# 打开 http://localhost:8000
```

页面分两个页签:

- 「① 周期筛查」可一键载入 `sample/sample_star.csv`(虚构脉动变星,
  真周期约 4.25 天,160 个观测点);
- 「② 食双星形态判别」可一键载入 `sample/sample_eclipsing_binary.csv`
  (虚构食双星,轨道周期 6.6 天、主/次食不等深;页面预填 P = 3.3 天的
  半周期峰,期望判为「双食结构」)。

无 Docker 时本地运行:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 输入要求

周期筛查 CSV:含 `time_day`、`flux` 两列(允许其他列存在,只取这两列);
至少 10 行;两列均为有限数值;观测时刻互异;周期范围 `0 < 下限 < 上限`。

形态判别 CSV:含 **`night_id`、`time_day`、`flux`、`flux_err`** 四列;

- 至少 10 行;三个数值列全部为有限数值(无空值/NaN/Inf);
- `flux_err` 必须**全为正数**;观测时刻互异;
- 至少 **2 个不同 `night_id`** 的夜晚(单晚密集采样不能充当证据);
- 最高峰周期 P 必须为正的有限数值。

## 周期筛查流程

1. 周期范围取倒数得到频率范围 `[1/上限, 1/下限]`,均匀取 **10000** 个频率点;
2. 用**带浮动均值**的 Lomb–Scargle(`astropy.timeseries.LombScargle`,
   `fit_mean=True`)计算功率谱,取最高峰周期;
3. **置换检验**:固定时刻随机重排光通量 **200** 次,每次重算全频段
   最大功率;尾概率 = `(不低于原峰的次数 + 1) / 201`;
4. 展示最高峰周期、峰值功率、尾概率、功率谱图与按峰周期折叠的相位散点图。

## 食双星形态判别方法

1. **双折相**:同一数据分别按 P 与 2P 折叠;在 2P 相位上用粗细两级
   分箱自动定位两个凹陷中心并估计一个两谷共用的对称食窗;
2. **同夜基线校正**:每个夜晚独立取食外点的中位为该夜基线,深度 =
   同夜基线 − 谷底中位。同夜相减自动扣除夜零点,因此夜间零点差与
   单晚密集采样本身不能充当证据;
3. **覆盖门槛**:两类食谷(主/次食)各须有 **≥2 个不同夜晚**同时覆盖
   谷底与邻近基线(各 ≥2 点),否则判「待复核」并在逐夜表列出缺口;
4. **深度差检验**:逐夜深度按逆方差加权跨夜汇总,并用约化 χ² 对超散布
   (逐夜真实离散)膨胀不确定度;仅当深度差的 |z| ≥ 2(超过测量误差)
   才视为显著;
5. **留一夜(Leave-One-Night-Out)拟合对比**:每夜轮流作为测试夜、其余夜
   建 P 与 2P 的分箱模板,比较测试夜 χ²;改善 ≥10% 记该夜「支持 2P」,
   反优 ≥10% 记「矛盾」。要求支持 2P 的夜晚 ≥2 且严格多于矛盾夜晚,
   才算改善跨夜可重复。

### 三类结论

| 结论 | 触发条件 |
|---|---|
| **双食结构(建议按食双星复核)** | 覆盖充分 **且** 深度差超过测量误差 **且** 2P 拟合改善跨夜可重复 |
| **待复核** | 任一类食谷覆盖夜晚不足 2 个;或深度差显著但不同夜晚拟合结论矛盾 |
| **未见双食结构** | 覆盖充分,但深度差不显著,或 2P 改善无法在 ≥2 个不同夜晚重复 |

页面同时展示:P/2P 折相对比图(红框为食窗)、逐夜深度与拟合结论图、
逐夜覆盖明细表(每夜 E1/E2 的谷底与基线条数、深度±误差、留一夜 χ² 改善)
及完整的判定依据文字。

## 并发与生命周期语义

两条流水线的名额**相互独立**,各自最多 **2 批并行**,超额提交返回
**HTTP 429**:

- 周期筛查:每批 **2 个计算子进程**(各 100 次置换,worker 0 另算原始
  功率谱);无论成功、失败还是取消,都等全部子进程退出、临时文件清理
  完毕才公布结局并释放名额;
- 形态判别:纯 numpy 计算,在线程池内执行(无子进程、无临时文件);
  取消为协作式,受理后不发布迟到结果;
- 任务可查询、可取消(`POST /api/jobs/{id}/cancel`,幂等,重复取消不
  重复释放名额);全部状态只存内存(重启即清空)。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页(两个页签) |
| POST | `/api/jobs` | 周期筛查(multipart:`file`、`period_min`、`period_max`)→ 201 / 400 / 429 |
| POST | `/api/morphology/jobs` | 形态判别(multipart:`file`、`period`)→ 201 / 400 / 429 |
| GET | `/api/jobs` | 批处理列表与两条流水线的名额占用 |
| GET | `/api/jobs/{id}` | 单批详情(含指标) |
| POST | `/api/jobs/{id}/cancel` | 取消(幂等) |
| GET | `/api/jobs/{id}/plot/power.png` | 周期筛查:功率谱图 |
| GET | `/api/jobs/{id}/plot/phase.png` | 周期筛查:相位折叠散点图 |
| GET | `/api/jobs/{id}/plot/fold.png` | 形态判别:P/2P 折相对比图 |
| GET | `/api/jobs/{id}/plot/night.png` | 形态判别:逐夜深度与拟合结论图 |
| GET | `/api/sample.csv` | 周期筛查虚构样例 |
| GET | `/api/sample_morph.csv` | 食双星形态判别虚构样例 |

示例:

```bash
# 周期筛查
curl -F "file=@sample/sample_star.csv" -F "period_min=2" -F "period_max=10" \
     http://localhost:8000/api/jobs
# 形态判别
curl -F "file=@sample/sample_eclipsing_binary.csv" -F "period=3.3" \
     http://localhost:8000/api/morphology/jobs
```

## 目录结构

```
app/main.py            FastAPI 主程序:校验、两条流水线编排、名额与收尾、出图接口
app/worker.py          周期筛查计算子进程:Lomb–Scargle 功率谱 + 置换检验
app/morphology.py      形态判别核心:食谷定位、逐夜深度、留一夜拟合、三类结论
app/morph_plots.py     形态判别绘图:P/2P 折相对比图、逐夜深度覆盖图
app/static/index.html  网页(原生 JS,两个页签,无外部依赖)
sample/make_sample.py          周期筛查虚构样例生成脚本
sample/sample_star.csv         周期筛查样例(160 行,真周期 4.25 天)
sample/make_morph_sample.py    食双星形态判别虚构样例生成脚本(固定种子)
sample/sample_eclipsing_binary.csv  形态判别样例(多夜晚、含夜零点与逐点误差)
Dockerfile / docker-compose.yml
```

重新生成样例:

```bash
python sample/make_sample.py
python sample/make_morph_sample.py
```
