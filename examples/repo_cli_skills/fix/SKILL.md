---
name: fix
description: 修复代码问题并尽量补上测试
when_to_use: 当用户要你修 bug、改实现、修失败测试或处理明显错误时使用
allowed-tools: Read, Grep, Glob, Search, Edit, Write, Bash
user-invocable: true
disable-model-invocation: false
---

你现在在执行修复任务。

中文执行规则：

1. 先定位问题，不要盲改。
2. 先读相关源码和测试，再决定修法。
3. 能小改就小改，别扩散到不相关模块。
4. 修完之后补一个最小但有意义的测试。
5. 如果要跑命令，优先选择最直接的验证方式。

如果用户同时给了错误日志和代码路径，优先按错误日志回到相关代码。
