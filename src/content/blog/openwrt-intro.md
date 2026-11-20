---
title: Openwrt 记录
description: Openwrt 记录
pubDatetime: 2026-11-20T22:10:00+08:00
tags:
  - 网络
featured: false
draft: false
---

# Openwrt 记录

## 系统路由

一个数据包达到系统之后，系统路由决定如何处理这个数据包。

Linux 中的 `ip` 是一个**网络配置与管理工具（iproute2 软件包的一部分）**，主要用于管理 Linux 内核中的网络协议栈。

它不是一个独立的网络服务，而是一个用户空间命令行程序，通过 Netlink 接口与 Linux 内核通信，修改或查询内核中的网络配置。

### `ip` 属于哪个软件包？

| 项目     | 内容                          |
| ------ | --------------------------- |
| 软件包    | `iproute2`                  |
| 命令     | `/sbin/ip` 或 `/usr/sbin/ip` |
| 主要维护者  | iproute2 开发社区               |
| 内核交互接口 | Netlink                     |
| 配置对象   | 网卡、IP 地址、路由、策略路由、隧道等        |


### `ip` 能管理什么？

`ip` 实际上是一个功能集合，主要包括：

| 子命令         | 功能             | 示例               |
| ----------- | -------------- | ---------------- |
| `ip link`   | 管理网络接口         | `ip link show`   |
| `ip addr`   | 管理 IP 地址       | `ip addr show`   |
| `ip route`  | 管理路由表          | `ip route show`  |
| `ip rule`   | 管理策略路由规则       | `ip rule show`   |
| `ip neigh`  | 管理 ARP/NDP 邻居表 | `ip neigh show`  |
| `ip tunnel` | 管理隧道           | `ip tunnel show` |
| `ip netns`  | 管理网络命名空间       | `ip netns list`  |

尤其是 `ip rule` 与 `ip route`：前者决定内核根据什么条件选择路由表，后者决定在选定的路由表中如何找到下一跳。这也是理解 OpenWrt 透明代理、策略路由和流量分流的基础。

### `ip rule` 策略路由规则

执行 `ip rule show`：

```
0:      from all lookup local
32766:  from all lookup main
32767:  from all lookup default
```

这些规则按照优先级（数字越小越先执行）依次匹配。规则可以根据：源 IP 地址、目标 IP 地址、入接口、fwmark（防火墙标记）、TOS 等选择不同的路由表。

例如 `0:      from all lookup local` 优先级是 0，匹配任何源 IP，根据 local 路由表进行转发。

### `ip route` 路由表

执行 `ip route show table all`：

```
default via 192.168.5.1 dev br-lan
192.168.5.0/24 dev br-lan scope link
```
假设数据包目标是 8.8.8.8，内核会匹配 default 路由，于是：下一跳：192.168.5.1，出口接口：br-lan。如果存在多个匹配路由，通常选择最长前缀匹配（Longest Prefix Match），策略路由则会影响使用哪张路由表。

一般格式：

```
[路由类型] 目标网络
    [via 下一跳网关]
    [dev 出口网卡]
    [proto 路由来源]
    [scope 路由作用范围]
    [src 源地址]
    [table 路由表]
    [metric 优先级]
```

常见接口：

| 接口       | 含义         |
| -------- | ---------- |
| `eth0`   | 以太网物理接口    |
| `br-lan` | LAN 网桥     |
| `wan`    | WAN 逻辑接口   |
| `lo`     | 本地环回接口     |
| `tun0`   | TUN 虚拟网络接口 |

proto：路由来源

常见值：

| proto     | 含义               |
| --------- | ---------------- |
| `kernel`  | Linux 内核自动生成     |
| `static`  | 静态路由             |
| `dhcp`    | DHCP 获取的路由       |
| `boot`    | 启动时配置的路由         |
| `186` 等数字 | 某些用户空间程序自定义的协议编号 |

scope：路由作用范围

`scope link` 表示目标网络在本地链路上，可以直接通过该接口访问，不需要经过网关。

常见值：

- `host`：仅本机有效。
- `link`：本地链路范围。
- `global`：全局范围，通常默认不显示。

src：首选源 IP

```
src 192.168.5.2
```

表示通过这条路由发送数据时，优先使用 `192.168.5.2` 作为源 IP 地址。

特殊路由类型，Linux 路由表还可以包含特殊路由：

| 类型            | 含义                 |
| ------------- | ------------------ |
| `local`       | 目标地址属于本机           |
| `broadcast`   | 广播路由               |
| `unicast`     | 普通单播路由（默认类型）       |
| `blackhole`   | 丢弃数据包              |
| `unreachable` | 丢弃并返回不可达错误         |
| `prohibit`    | 禁止访问，返回管理性禁止错误     |
| `throw`       | 终止当前路由表查找，继续后续策略规则 |

例如：

```
blackhole 10.0.0.0/8
```

表示发往 `10.0.0.0/8` 的数据包直接丢弃。

如何看出路由属于哪张表？


例如：

```
default via 192.168.5.1 dev br-lan
default dev tun0 table 354
```

第一条没有指定 `table`，默认属于 `main` 表。

第二条明确指定：

```
table 354
```

因此属于编号为 354 的路由表。

可以分别查看：

```
ip route show table main
ip route show table local
ip route show table default
ip route show table 354
```

因此，我们可以建立这样的对应关系：

**Linux 路由决策流程**

1. 数据包进入 Linux 内核
2. 检查 local 路由表
3. **检查 fwmark 是否为 0x162**
4. 匹配
5. 不匹配则继续
6. 查询 table 354
7. 查询 main 路由表
8. 查询 default 路由表

还有一个非常实用的命令：`ip route get 8.8.8.8` 它可以模拟内核对某个目标地址进行路由查询，显示最终选择的出口网卡、下一跳和源 IP。

```
ip route get 8.8.8.8 # 外网 IP
8.8.8.8 via 192.168.5.1 dev br-lan src 192.168.5.2 uid 0 # 经过 192.168.5.1 网关
```

```
ip route get 127.0.0.1 # 本机
local 127.0.0.1 dev lo table local src 127.0.0.1 uid 0  # lo 本机
```

```
ip route get 192.168.5.3 # 局域网地址
192.168.5.3 dev br-lan src 192.168.5.2 uid 0  # 不经过网关直接通过 br-lan 发往目标
```

### 转发链

Linux 实际上有三条不同的数据包处理路径：

| 路径      | Netfilter 链                        | 作用         |
| ------- | ---------------------------------- | ---------- |
| INPUT   | PREROUTING → INPUT                 | 数据包交给本机    |
| FORWARD | PREROUTING → FORWARD → POSTROUTING | 数据包经过本机转发  |
| OUTPUT  | OUTPUT → POSTROUTING               | 本机程序产生的数据包 |

```
sysctl net.ipv4.ip_forward # 决定是否开启 FORWARD 链路
net.ipv4.ip_forward = 0
```

## nftables 系统 Netfilter 管理工具

Netfilter 链路处理过程中，包含五个 hook：

| 阶段          | 发生的事情                        |
| - | - |
| PREROUTING | conntrack、DNAT、路由决策之前的处理，主要是修改目标 IP 端口 |
| INPUT | 数据包到达本机程序 |
| FORWARD | 数据包转发，上一节说的 `ip` 规则工作在这里 |
| OUTPUT | 数据包由本机程序发送 |
| POSTROUTING | SNAT、MASQUERADE 路由后处理，主要是修改源目标 IP 端口 |

执行 `nft list ruleset` 查看所有规则链路，旧版由 `iptables` 工具提供。

它的结构大致是：

```
table <地址族> <表名> {
    chain <链名> {
        type <链类型> hook <钩子> priority <优先级>;
        policy <默认策略>;

        <匹配条件> <执行动作>
    }
}
```

常见地址族：

| 地址族      | 含义               |
| -------- | ---------------- |
| `ip`     | IPv4             |
| `ip6`    | IPv6             |
| `inet`   | 同时处理 IPv4 和 IPv6 |
| `arp`    | ARP              |
| `bridge` | 网桥数据包过滤          |
| `netdev` | 网络设备入口处理         |

链是规则的集合，数据包经过对应的 Hook 时，会按照规则顺序进行匹配和处理。


例如：

```
type nat hook postrouting priority srcnat; policy accept;
```

这几个字段非常重要：

| 字段                 | 含义                 |
| ------------------ | ------------------ |
| `type nat`         | NAT 类型的链           |
| `hook postrouting` | 在 POSTROUTING 阶段执行 |
| `priority srcnat`  | 在源 NAT 优先级执行       |
| `policy accept`    | 默认策略为 ACCEPT       |

常见 Hook：

- `prerouting`
- `input`
- `forward`
- `output`
- `postrouting`

具体规则：

例如：

```
oifname "eth0" masquerade
```

意思是：

如果数据包从 `eth0` 网卡发出，就执行 MASQUERADE。

再看：

```
ct state established,related accept
```

表示连接状态为 established 或 related 时，允许数据包通过。

可以把整个结构理解成：

**Table：ip nat**

规则表

**Chain：postrouting**

数据包处理链

**Hook：postrouting**

内核调用链的时机

**Rule：oifname "eth0" masquerade**

匹配条件与执行动作

## NAT 规则

### SNAT

假设：

| 设备          | IP            |
| ----------- | ------------- |
| 内网客户端       | 192.168.5.100 |
| ImmortalWrt | 192.168.5.2   |
| 外部服务器       | 8.8.8.8       |

客户端发送 TCP SYN：

```
192.168.5.100:50000 → 8.8.8.8:443
```


处理过程：

| 阶段          | 发生的事情                        |
| ----------- | ---------------------------- |
| PREROUTING  | conntrack 创建连接记录             |
| 路由决策        | 查询 ip rule / ip route        |
| FORWARD     | 检查防火墙是否允许                    |
| POSTROUTING | 执行 SNAT，将源 IP 改成 192.168.5.2 |
| 网卡发送        | 发往外部服务器                      |

此时 conntrack 记录了原始连接和 NAT 映射。


例如：

```
Original:
192.168.5.100:50000 → 8.8.8.8:443

Translated:
192.168.5.2:50000 → 8.8.8.8:443
```

这里假设没有发生源端口变化。

注意：SNAT 并不是把连接记录建立在某个独立的 TCP 转发表里，而是利用 conntrack 保存连接状态和 NAT 映射。

现在外部服务器返回 SYN-ACK：


```
8.8.8.8:443 → 192.168.5.2:50000
```

这是问题的重点。

返回数据包进入 ImmortalWrt 后：

**第一阶段：PREROUTING**

conntrack 识别出这是已有连接的返回方向。

根据之前建立的 NAT 映射，执行反向 SNAT。

原始数据包：

```
8.8.8.8:443 → 192.168.5.2:50000
```

反向 NAT 后：

```
8.8.8.8:443 → 192.168.5.100:50000
```

这里虽然叫反向 SNAT，但从当前数据包的角度看，实际修改的是目的地址，因此也可以理解为自动执行了目的地址恢复。

**第二阶段：路由决策**

现在 Linux 进行路由查询时，看到的是：

```
Destination: 192.168.5.100
```

而不是：

```
Destination: 192.168.5.2
```

于是内核根据 `ip rule` 和 `ip route` 查找通往 `192.168.5.100` 的路由。

假设：

```
192.168.5.0/24 dev br-lan
```

那么数据包就会从 `br-lan` 发出。

**第三阶段：FORWARD**

数据包经过防火墙。

由于 conntrack 已经识别出这是已有连接的返回方向，其连接状态通常为：

```
ESTABLISHED
```

如果防火墙允许：

```
ct state established,related accept
```

数据包就会被放行。

**第四阶段：POSTROUTING**

完成后续处理，从 LAN 接口发送给：

```
192.168.5.100
```

整个过程没有绕过路由系统。

