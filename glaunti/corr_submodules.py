import jax
import jax.numpy as jnp
import jax.image
import equinox as eqx

from typing import Tuple, Callable


class Conv2dINAct(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    act: Callable = eqx.field(static=True) 

    def __init__(self, in_ch, out_ch, key, act=jax.nn.swish):
        self.conv = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding="SAME", key=key)
        self.norm = eqx.nn.GroupNorm(groups=out_ch, channels=out_ch)
        self.act = act

    def __call__(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DoubleConv2d(eqx.Module):
    block1: Conv2dINAct
    block2: Conv2dINAct

    def __init__(self, in_ch, out_ch, key):
        k1, k2 = jax.random.split(key, 2)
        self.block1 = Conv2dINAct(in_ch, out_ch, key=k1)
        self.block2 = Conv2dINAct(out_ch, out_ch, key=k2)

    def __call__(self, x):
        return self.block2(self.block1(x))


class Down(eqx.Module):
    block: DoubleConv2d
    down: eqx.nn.MaxPool2d = eqx.field(static=True) 

    def __init__(self, in_ch, out_ch, key):
        self.block = DoubleConv2d(in_ch, out_ch, key=key)
        self.down = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x):
        x = self.block(x)
        skip = x
        x = self.down(x)
        return x, skip


class Up(eqx.Module):
    block: DoubleConv2d

    def __init__(self, in_ch, out_ch, key):
        self.block = DoubleConv2d(in_ch, out_ch, key=key)

    def __call__(self, x, skip):
        c, h, w = x.shape
        x = jax.image.resize(
            x, shape=(c, h * 2, w * 2), method="bilinear", antialias=False
        )
        x = jnp.concatenate([skip, x], axis=0)
        return self.block(x)


class Conv1dINAct(eqx.Module):
    conv: eqx.nn.Conv1d
    norm: eqx.nn.GroupNorm
    act: Callable = eqx.field(static=True) 

    def __init__(self, in_ch, out_ch, key, act=jax.nn.swish):
        self.conv = eqx.nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding="SAME", key=key)
        self.norm = eqx.nn.GroupNorm(groups=out_ch, channels=out_ch)
        self.act = act

    def __call__(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DoubleConv1d(eqx.Module):
    block1: Conv1dINAct
    block2: Conv1dINAct

    def __init__(self, in_ch, out_ch, key):
        k1, k2 = jax.random.split(key, 2)
        self.block1 = Conv1dINAct(in_ch, out_ch, key=k1)
        self.block2 = Conv1dINAct(out_ch, out_ch, key=k2)

    def __call__(self, x):
        return self.block2(self.block1(x))


class Corrector2dBranch(eqx.Module):
    in_conv: eqx.nn.Conv2d
    downs: Tuple[Down, ...]
    bottleneck: DoubleConv2d
    ups: Tuple[Up, ...]
    n_stages: int = eqx.field(static=True)

    def __init__(self, input_size, n_filters, n_stages, key):
        self.n_stages = n_stages
        
        keys = jax.random.split(key, 2 * n_stages + 2)
        keys = iter(keys)

        self.in_conv = eqx.nn.Conv2d(input_size, n_filters, kernel_size=1, stride=1, padding="SAME", key=next(keys))
        
        downs = []
        for _ in range(n_stages):
            downs.append(Down(n_filters, n_filters, key=next(keys)))
        self.downs = tuple(downs)

        self.bottleneck = DoubleConv2d(n_filters, n_filters, key=next(keys))

        ups = []
        for _ in range(n_stages):
            ups.append(Up(2 * n_filters, n_filters, key=next(keys)))
        self.ups = tuple(ups)

    def __call__(self, x, initial_h=None, return_series=False):
        y = self.in_conv(x)

        skips = []
        for down in self.downs:
            y, skip = down(y)
            skips.append(skip)

        y = self.bottleneck(y)

        for up, skip in zip(self.ups, reversed(skips)):
            y = up(y, skip)

        return y


class Corrector1dBranch(eqx.Module):
    in_conv: eqx.nn.Conv1d
    conv_blocks: Tuple[DoubleConv1d, ...]
    pools: Tuple[eqx.nn.AvgPool1d, ...] = eqx.field(static=True) 
    n_stages: int = eqx.field(static=True)

    def __init__(self, input_size, n_filters, n_stages, key):
        self.n_stages = n_stages
        
        keys = jax.random.split(key, n_stages + 1)
        keys = iter(keys)

        self.in_conv = eqx.nn.Conv1d(input_size, n_filters, kernel_size=1, stride=1, padding="SAME", key=next(keys))
        
        conv_blocks, pools = [], []
        for _ in range(n_stages):
            conv_blocks.append(DoubleConv1d(n_filters, n_filters, key=next(keys)))
            pools.append(eqx.nn.AvgPool1d(kernel_size=2, stride=2))
        self.conv_blocks = tuple(conv_blocks)
        self.pools = tuple(pools)

    def __call__(self, x, initial_h=None, return_series=False):
        y = self.in_conv(x)

        for conv_block, pool in zip(self.conv_blocks, self.pools):
            y = conv_block(y)
            y = pool(y)

        # global pool
        y = jnp.mean(y, axis=-1, keepdims=True)

        return y
