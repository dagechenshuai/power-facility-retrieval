# 开发约定

Python 命令使用 `uv run`。修改后执行 `uv run pytest -q`，不得通过跳过失败测试宣称通过。

- 查询循环及边界见 `docs/standards/query-decisions.md`。
- 外部依赖与运行步骤见 `docs/standards/running.md`。
- 评测资料及历史指标边界见 `docs/evaluation.md`。
- 不提交 `.env`、真实凭据或无权公开的企业资料。
- 模型替身和合成资料必须注明用途，不能当作真实模型效果或历史实验结果。
