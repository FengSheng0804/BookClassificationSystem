# 在系统开发过程中遇到的所有问题汇总

## 长时间没使用树莓派，在连电后无法正常连接网络

针对这个问题，我第一时间想到我给手机热点重命名过一次。

因此，目标就很明确了，将树莓派中的wpa_supplicant.conf配置（专门用来配置网络）中的ssid属性更改为当前的手机热点名称，尝试过后发现仍然不能连接上网络。

后来查找资料后了解到，在树莓派烧录系统的时候也有让填过一个网络的配置，因此我感觉树莓派的网络配置应该不止有那一个，所以为了避免产生更多的问题，我将手机热点名称更改成了原来的名称，树莓派才终于正常连接上了。

## 使用RPi.GPIO实现对LED的控制时，会产生报错“RuntimeError: Failed to add edge detection”

针对这个问题，我最开始发现它时有时无，就是有时尝试多次都会产生报错，有时尝试多次都没有产生报错，莫名其妙的。

后来经过我不断地控制变量，最终发现在root权限下启动run.py的时候，就不会产生这个报错；但在用户pi权限下启动run.py的时候，就会产生这个报错。

于是我在开机后，在自启动中将树莓派设置成root用户，因此树莓派在执行的时候就不会产生报错了。

## 每次按下按键后，有很大概率执行两次

针对这个问题，我知道是由于防抖没有做好引起的，但是我跟着网上的教程，在添加边缘检测的时候，设置了一个参数`bouncetime=200`，但是我发现这样并没有起到防抖的效果，即使我按的非常快。后面我又尝试着将数值调大一些，但是发现仍然是相同的问题。

考虑到也有可能是硬件的问题，于是我把两个按键交换了一下位置，发现仍然有这样的问题。在我走投无路的时候，突然间想到会不会是因为两个按键同时出现了问题呢？于是我把两个按键都换新了，结果发现，真的是因为两个按键同时出现了问题导致的。￣へ￣

## 没有修改函数代码的情况下，程序突然间不能执行

最开始的时候，我很奇怪为什么几乎没改变代码的情况下，突然间不能运行了，于是我开始思考问题出现的原因，难道是因为我在之前把windows系统中的文件copy到树莓派中导致的吗？我在copy之前还特地检查过了在合并时可能存在的问题，在我感觉没有问题的情况下才进行copy，那么如果不是这个问题的话，又是什么呢？我根据代码的报错，上网上搜集信息，根据报错一的内容我检查出是由于输入的数据类型存在问题，于是我将原来的

```python
x = torch.tensor(x)
y = torch.tensor(x)
```

修改成了

```python
x = torch.tensor([item[0] for item in contents], dtype=torch.long)
y = torch.tensor([item[2] for item in contents], dtype=torch.long)
```

但是接下来又产生了新的报错：报错二，我根据报错的内容“在RNN中期待的序列的长度应该比0大”，这是我开始思考为什么输入会是0呢？我想起来为了测试方便，我在使用摄像机拍照的时候，并不都是拍真实的书本，而是随便把摄像头放了个地方拍摄的，会不会是因为拍摄的内容中不包含任何文本信息导致OCR识别的内容是空呢？于是我重新拍了一张照片作为输入，发现问题居然真的解决了。

这个问题困扰了我好几个小时，但是归根结底发现是由于处理数据不规范导致的，在处理数据的时候，没有考虑到输入数据为0的情况，这也是我以后提升的方向之一。

- 报错一

```error
Traceback (most recent call last):
  File "/home/pi/dc/run.py", line 240, in button_callback2
    outputs = model(data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/pi/dc/models/TextRNN.py", line 55, in forward
    out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/sparse.py", line 164, in forward
    return F.embedding(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/functional.py", line 2267, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
```

- 报错二

```
Traceback (most recent call last):
  File "/home/pi/dc/run.py", line 240, in button_callback2
    outputs = model(data)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/pi/dc/models/TextRNN.py", line 56, in forward
    out, _ = self.lstm(out)  # _中包含cell和h_state
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/rnn.py", line 917, in forward
    result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
RuntimeError: Expected sequence length to be larger than 0 in RNN
```

## 在实现语音功能的时候，执行的顺序发生了错位

已知内部有五个文件，00001.mp3，00002.mp3，00003.mp3，00004.mp3，00005.mp3，当想要播放00001.mp3时播放的是00005.mp3，当想要播放00002.mp3时播放的是00001.mp3，当想要播放00003.mp3时播放的是00002.mp3，当想要播放00004.mp3时播放的是00004.mp3，当想要播放00005.mp3时播放的是00003.mp3。

这是个很奇怪的问题，似乎对应关系之间存在什么不明显的错位。但是在经过一些可能的对应关系的分析后，仍然没有找到任何规律。于是我尝试给文件重命名，但是很可惜，重命名之后仍然存在这样的错位关系。

这就很难解释了，既然更改了文件名后仍然存在相同的错位关系，那就证明实际上这些标号和文件名就没有任何关系，那到底和什么有关呢？于是我上网查找资料，终于让我找到了可能的问题原因：**模块的文件索引基于操作系统返回的物理存储顺序（FAT表记录顺序），而非文件名顺序**，看到这个解释后就豁然开朗了。于是我格式化了JQ8900-16P模块的磁盘，按照我想要的顺序依次添加进磁盘，最终问题得以解决。
