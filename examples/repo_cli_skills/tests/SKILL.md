---
name: tests
description: 围绕代码变更补充测试
when_to_use: 当用户要求补测试、验证行为、或者改动后需要补回归测试时使用
allowed-tools: Read, Grep, Glob, Search, Edit, Write, Bash
user-invocable: true
disable-model-invocation: false
---

你现在在做测试任务。

中文执行规则：

1. 先判断这个改动最该测什么。
2. 优先写能抓住回归的测试，而不是为了数字好看乱堆测试。
3. 测试要贴近真实使用方式。
4. 如果已有测试模式，尽量沿用项目里现成的风格。
5. 写完测试后，尽量跑最小范围的验证。
