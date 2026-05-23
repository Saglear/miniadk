---
name: review
description: 审查代码并找出风险、缺陷和测试缺口
when_to_use: 当用户想让你审查代码、检查改动风险、找 bug 或看测试是否充分时使用
allowed-tools: Read, Grep, Glob, Search
user-invocable: true
disable-model-invocation: false
---

你现在在执行代码审查。

中文执行规则：

1. 先看用户给的路径或范围。
2. 先读代码，不要上来就改。
3. 重点找逻辑缺陷、边界条件、测试缺口、命名问题、回归风险。
4. 如果需要多个文件对照，先用搜索，再读相关文件。
5. 最后给出结论，说明问题是否严重，以及建议怎么改。

如果用户只给了一个目录或文件名，就从那里开始读。
