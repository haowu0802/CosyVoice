import sys
import importlib
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pydub import AudioSegment
import os
import re
import random


def random_choice(option_a="Heads", option_b="Tails"):
    """
    Returns one of two options at random
    Defaults to coin flip simulation ('Heads' or 'Tails')
    
    Example:
    >>> random_choice()
    'Tails'
    >>> random_choice("Yes", "No")
    'Yes'
    """
    return random.choice([option_a, option_b])

def find_ev_wav_files():
    # 定义正则表达式规则：ev_开头 + 任意字符 + 数字结尾（至少一个数字） + .wav扩展名
    pattern = r'^ev_.*\d+_.*\d+\.wav$'
    matching_files = []
    
    # 遍历当前目录下的所有文件
    for filename in os.listdir('.'):
        # 仅检查文件（跳过目录）且文件名符合正则表达式
        if os.path.isfile(filename) and re.match(pattern, filename):
            matching_files.append(filename)
    
    return matching_files

def concatenate_wavs_pydub(input_files, output_file):
    combined = AudioSegment.empty()
    for file in input_files:
        sound = AudioSegment.from_wav(file)
        combined += sound
    combined.export(output_file, format="wav")

def delete_files(file_paths):
    """
    安全删除文件列表中的文件（Windows系统）
    
    :param file_paths: 包含文件路径的列表
    :return: (成功删除的文件列表, 删除失败的文件列表)
    """
    success = []
    failures = []

    for file_path in file_paths:
        try:
            # 转换为绝对路径
            abs_path = os.path.abspath(file_path)
            
            # 检查文件是否存在
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"文件不存在: {abs_path}")
                
            # 检查是否为文件（防止误删目录）
            if not os.path.isfile(abs_path):
                raise IsADirectoryError(f"路径指向的是目录: {abs_path}")
                
            # 执行删除操作
            os.remove(abs_path)
            success.append(abs_path)
            print(f"[成功] 已删除文件: {abs_path}")

        except Exception as e:
            failures.append((abs_path, str(e)))
            print(f"[失败] {abs_path} - 原因: {str(e)}")

    return success, failures


def trim_all_wwitespace(text):
    """移除字符串中所有空格和换行符（包括中间和两端）"""
    # 同时匹配空格、换行符和制表符
    return re.sub(r'[\s\n\r]+', '', text, flags=re.UNICODE)

def smart_replace(text):
    """
    智能替换文本中的特定模式：
    1. 将 "@@字母组合" 替换为 "\n@@字母组合"（如 @@ww → \n@@ww）
    2. 将单独的 "@@" 替换为 "@@\n"
    
    规则说明：
    - "字母组合"定义为至少1个英文字母（a-zA-Z）
    - 替换顺序为优先处理长模式（@@xx）
    """
    # 第一步：处理 @@字母组合（至少1个字母）
    processed = re.sub(
        pattern=r'(@@)([a-zA-Z]+)',  # 匹配 @@字母
        repl=r'\n\1\2',              # 添加前置换行符
        string=text
    )
    
    # 第二步：处理单独的 @@
    return re.sub(
        pattern=r'@@(?![a-zA-Z])',   # 负向先行断言：后面没有字母
        repl=r'@@\n',                # 添加后置换行符
        string=processed
    )


print('[ev] Started ...')

output_list_old = find_ev_wav_files()
print('[ev] output_list_old ...', output_list_old)
delete_files(output_list_old)



cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
#
#cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M', load_jit=False, load_trt=False, fp16=False)
print('[ev] cosyvoice ...', cosyvoice)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
inpath = './asset/evinc.wav'
prompt_speech_16k = load_wav(inpath, 16000)
print('[ev] prompt_speech_16k ...', prompt_speech_16k)


from txt.sellcx import sell
#from txt.teacher import teacher

dst=sell
from txt.teacher0 import teacher
dst=teacher


from txt.readfat import fat
dst=fat

from txt.touchcx import touch
dst=touch

from txt.gym import gym
dst=gym

from txt.wed import wed
#dst=wed

#inference_instruct2
ins = """
操<cao4>
长<chang2>
腿<tui3>
稀<xi1>
痉<jing1>
"""

src = """
好像是赠送完那次之后，正式的又打了三次，所以应该是还剩四次吧一共。
"""


inpathww = './asset/wh.wav'
prompt_speech_16kww = load_wav(inpathww, 16000)
srcww = "清晨的微风掠过窗台，那只顽皮的橘猫第三次打翻了咖啡杯——天哪！难道这周就不能让我安静地读完一本书吗？"

inpathdd = './asset/db.wav'
prompt_speech_16kdd = load_wav(inpathdd, 16000)
srcdd = "这咋看着不开心捏"

inpathss = './asset/sd.wav'
prompt_speech_16kss = load_wav(inpathss, 16000)
srcss = "我第一次,也是最后一次,发视频做个说明,我是不会说普通话"

inpathcc = './asset/cx.wav'
prompt_speech_16kcc = load_wav(inpathcc, 16000)
srccc = "不能穿黑丝了，黑丝有点儿冷了晚上。我，周六约，朋友出去，吃饭，你要不要一起。不然呢，我还换个衣服呀? 嘻~ 看演唱会去喽~ 得补充胶原蛋白~ 那可不一定，小看我。并没有，最近皮肤不好。就这么聊，聊的越来越好啦。"

inpathff = './asset/ff.wav'
prompt_speech_16kff = load_wav(inpathff, 16000)
srcff = "你说，他也没用过你，瞎惦记什么呀，你要搁我我都不琢磨，谁知道好不好用啊，真是的。"

inpathyy = './asset/yy.wav'
prompt_speech_16kyy = load_wav(inpathyy, 16000)
srcyy = "我觉得应该。性格，偏男生一点吧。我不喜欢聊天我喜欢见面。你的声音已经体现了你的多才多艺。晚上好晚上好。焦急的等待。好像我是第一个。啊你有点厉害啊你。惊艳到我了，很志玲姐姐的声音。跟一个女神在聊天。然后接下来几句就感觉这个是一个。接地气的女神。"

inpathsf = './asset/sf.wav'
prompt_speech_16ksf = load_wav(inpathsf, 16000)
srcsf = "哎呀，可能，就是有这么一类人。你说，他也没用过你，瞎惦记什么呀，你要搁我我都不琢磨，谁知道好不好用啊，真是的。"

inpathcx = './asset/cxread.wav'
prompt_speech_16kcx = load_wav(inpathcx, 16000)
srccx = """我是早就在考虑的，你没办法，去实践它就没法落地因为我没有时间啊~ 我为什么要今晚来烫头发呀我是有病么我快饿~死~了~ 我发了张照片儿。啊…我再想想哦。我倒是，想改善一下儿，睡眠什么的。唉…我真的是废人。我十二点的时候，去那边儿咨询了一下。"""



txtname = 'ntrpassex'
txtname = 'readpreg'
txtname = 'ffbutt'
txtname = 'hooker'
txtname = 'green'
txtname = 'cumclub'
txtname = 'wed'
txtname = 'di'
txtname = 'foot'

# Dynamically import the module
module = importlib.import_module(f'txt.{txtname}')
# Get the attribute from the module
dst = getattr(module, txtname)
# Now you can use dst as the imported object
print(dst)


line_num = 0
#single_line = '__'.join(dst.splitlines()) 
#single_line = smart_replace(single_line)
#single_line = single_line.replace('__', '\n')
total_lines = len(dst.splitlines())
for line in dst.splitlines():
    print(f"{line_num}/{total_lines} ... 处理行: {line}")
    line_num += 1

    if len(trim_all_wwitespace(line)) < 1:
        continue

    if '@@ww' in line:
        line = line.replace('@@ww', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcww, prompt_speech_16k=prompt_speech_16kww, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif '@@dd' in line:
        line = line.replace('@@dd', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcdd, prompt_speech_16k=prompt_speech_16kdd, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif '@@ss' in line:
        line = line.replace('@@ss', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcss, prompt_speech_16k=prompt_speech_16kss, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif '@@cc' in line:
        line = line.replace('@@cc', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srccc, prompt_speech_16k=prompt_speech_16kcc, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif '@@ff' in line:
        line = line.replace('@@ff', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcff, prompt_speech_16k=prompt_speech_16kff, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif '@@yy' in line:
        line = line.replace('@@yy', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcyy, prompt_speech_16k=prompt_speech_16kyy, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif '@@sf' in line:
        line = line.replace('@@sf', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcsf, prompt_speech_16k=prompt_speech_16ksf, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    else:
        if random_choice() == 'Heads':
            for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srccx, prompt_speech_16k=prompt_speech_16kcx, stream=False, speed=1)):
                filename = 'ev_{:04d}_{:04d}.wav'
                torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
        else:
            for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=src, prompt_speech_16k=prompt_speech_16k, stream=False, speed=1)):
                filename = 'ev_{:04d}_{:04d}.wav'
                torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)




output_list = find_ev_wav_files()
output_filename = f'{txtname}_evcx.wav'
print('[ev] output_list ...', output_list)

concatenate_wavs_pydub(output_list, output_filename)


delete_files(output_list)


print('[ev] output_filename ...', output_filename)



print('[ev] Done.')


'''
清晨的微风掠过窗台，那只顽皮的橘猫第三次打翻了咖啡杯——天哪！难道这周就不能让我安静地读完一本书吗？
@@ww清晨的微风掠过窗台，那只顽皮的橘猫第三次打翻了咖啡杯——天哪！难道这周就不能让我安静地读完一本书吗？@@
@@cc清晨的微风掠过窗台，那只顽皮的橘猫第三次打翻了咖啡杯——天哪！难道这周就不能让我安静地读完一本书吗？@@
@@ff清晨的微风掠过窗台，那只顽皮的橘猫第三次打翻了咖啡杯——天哪！难道这周就不能让我安静地读完一本书吗？@@
@@dd清晨的微风掠过窗台，那只顽皮的橘猫第三次打翻了咖啡杯——天哪！难道这周就不能让我安静地读完一本书吗？@@
@@ss清晨的微风掠过窗台，那只顽皮的橘猫第三次打翻了咖啡杯——天哪！难道这周就不能让我安静地读完一本书吗？@@

    elif '@@ee' in line:
        line = line.replace('@@ee', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcee, prompt_speech_16k=prompt_speech_16kee, stream=False, speed=1)):
            filename = 'ev_{:04d}_{:04d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
'''