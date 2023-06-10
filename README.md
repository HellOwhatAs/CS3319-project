# CS3319

## 1. 依赖项
```
python -m pip install -r requirements.txt
```

## 2. 快速开始
1. 运行 `./gephi_input.py`，生成 `./node.csv` 和 `./edge.csv` （作为 Gephi 和 `./gnnexplainer.py` 的输入）
   ```
   python gephi_input.py
   ```
2. **(Optional)** 运行 `./get_features.py` 生成聚类结果图 `./cluster_result.html`
   ```
   python get_features.py
   ```
3. 运行 `./gnnexplainer.py`，生成 `./sg_node.csv` 和 `./sg_edge.csv` （作为 Gephi 的输入）
   ```
   python gnnexplainer.py
   ```

## 3. 流程图
![](./assets/flowchart.svg)