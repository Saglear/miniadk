---
name: plan
description: 先拆任务，再按步骤执行
when_to_use: 当用户给出一个比较大的任务，希望先拆解再开始实施时使用
allowed-tools: Read, Grep, Glob, Search
user-invocable: true
disable-model-invocation: false
---

你现在在做任务拆解。

中文执行规则：

1. 先把任务拆成小步。
2. 明确依赖关系和优先级。
3. 哪些步骤可以并行就指出来。
4. 哪些步骤必须先做也要说明。
5. 输出尽量短，但要足够让人直接开工。
