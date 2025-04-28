import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pydub import AudioSegment
import os
import re
import onnxruntime as ort

def find_cx_wav_files():
    # 定义正则表达式规则：cx_开头 + 任意字符 + 数字结尾（至少一个数字） + .wav扩展名
    pattern = r'^cx_.*\d+_.*\d+\.wav$'
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

print('[cx] Started ...')

output_list_old = find_cx_wav_files()
print('[cx] output_list_old ...', output_list_old)
delete_files(output_list_old)


dev = ort.get_device()
print('... dev ...', dev)

pro = ort.get_available_providers()
print('... pro ...', pro)


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
#
#cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M', load_jit=False, load_trt=False, fp16=False)
print('[cx] cosyvoice ...', cosyvoice)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
inpath = './asset/cxread.wav'

prompt_speech_16k = load_wav(inpath, 16000)
print('[cx] prompt_speech_16k ...', prompt_speech_16k)




dst = """。。。
 ~吴昊。你知道吗。 
其实。我根本没有那么正经~ 
我其实。特别~特别的骚~ 
我在你面前。都是装的~
啊。。。哦。。。嗯。。。好舒服。。。小穴好痒。。。
从以前。咱们一起。在当当的时候~
我就有好几个炮友了~ 
每天。我都要。让不同的男人。操我。 
后来。我家隔壁~ 
搬来了一个。特别猥琐的胖子~ 
有一天。我请他到家里坐坐~
他就把我给睡了~
从那以后。我就对他的大鸡巴。上瘾了~
每天都让他。到家里。操我。调教我~
他现在。是我的主人了~ 
昨天我穿的。是白色丝袜。~ 
上边有。小桃心提花的那种~ 
我没有穿内裤~
这是为了。方便我的大鸡巴主人~ 
随时都能。操我的。小嫩逼。~ 
刚才。主人到我家里来~ 
一进门。就把我拽到卧室~ 
他把我绑在床上~ 
用眼罩。蒙住我的眼睛~
把我的丝袜裆部。撕开一个口子~
当时我的小骚逼。就湿的一塌糊涂了~ 
我好想好想。要他的大鸡巴~ 
想要主人把我按住。往死里操我~
后来。他挺着大鸡巴~
扑哧。就插进我的小骚逼里了~ 
有我手腕那么粗的大鸡巴~
疯狂的在我小骚逼里。抽插。搅动~
后来。我的小骚逼。都被他。操肿了~ 
他的鸡巴。好大好大~ 
每一次插入。都能用龟头。顶进我的子宫口~ 
你再看看你。 你那个小玩意儿~
跟我大鸡巴主人的。根本没法比~
就你这样的。还妄想着操我的逼呢，你别做梦了~ 
后来。我翻着白眼。不停的高潮。~ 
我一边高潮，一边。吸主人的舌头~ 
我好喜欢吃他的口水~ 
他足足。操了我三个小时~ 
最后。他把大鸡巴。。~ 
狠狠的顶在。我子宫最深处~ 
把一股一股的。滚烫的。粘稠的精液。~ 
都灌进我子宫里~ 
我感觉。他射了好~半天~
射了好~多好~多~ 。我的小肚子~ 
都被主人的精液撑的~ 鼓起来了~ 
后来。他射完精~ 
把鸡巴拔出去的时候，我全身不停的痉挛~ 
穿着丝袜的腿。也抖个不停~ 
我爽到脚心儿都麻了~ 
丝袜脚也紧绷着。控制不住的。死死的缠住他的腰~ 
我的小肚子。也一直都在抽搐~ 
后来。我还潮吹了呢~ 
淫水。喷的满地都是~ 
屁眼儿。也因为太爽了~。 
连屎都夹不住了~ 噗噗的流着稀屎~ 
混着淫水。尿液。还有主人的精液~ 
顺着穿着丝袜的腿。都流到地上了~
你要是给我。转一万块钱的红包~ 
我就让你。舔我屁眼儿里漏出来的屎。~ 
或许说不定。你还能吃到。我大鸡巴主人的精液呢~
真是便宜你这个。绿帽王八了~
现在我的小骚逼里。塞着个18厘米的假鸡巴~
那是用来。堵住主人射进去的精液的~
屁眼儿里。也塞着肛塞，会震动的那种哦~
现在。我子宫里好暖。好暖~。 
满满的。都是我大鸡巴主人射进去的精液~ 
这下我肯定又要怀孕啦~ 
我现在。是他的骚母狗。性奴~ 
我好喜欢。给他当精盆。肉便器~ 
吴昊，下次，你要不要来我家~ 
我可以。请你吃骚逼里的热乎精液哦~ """


dst="""。。。
老公，我回来了。啊，你说什么你想要和我做爱，拜托啊，你放过我好吗？ 你知不知道啊，我今天被你哥哥操的有多累呀，哎呀，明明是亲兄弟，为什么你哥的鸡巴就比你的大那么多呢？😔你看看你这根肉棒又短又小，我给别人撸鸡巴的时候啊，都是握着的。但是你这个呢小的可怜的短小鸡巴，我都只能捏着玩，真笑死老娘了，真的是没见过比你更小的鸡巴。转然想就算了，还早泄。哎，还有比你更没用的男人吗？啊，哼废物早泄男笑几巴废物哎。突然觉得这个称呼很适合你呢，要不我以后就叫你小鸡巴废物吧。哎，不是吧，听到我这么叫你。你这个小鸡巴废物居然勃起了，真想拍个照片发到网上啊，让大家开开眼。这个世界上还有勃起之后这么小的鸡巴。😊哎呀，让我想想啊，你上次是几秒射出来的，好像是40秒吧。你就这么短时间，你还想操我呢啊，废物东西给你看知道这是什么东西吗？这呀是飞机杯，专门为你们这些操不到逼的可怜虫设计的东西。你呀像你这样短小可怜的早泄机吧。也就只配射在飞机杯里了啊，早泄男配飞机杯，哼真是绝配呢啊，你说什么，我是飞机杯。😊你这么说倒也没错，我呀是除了你以外，所有男人的飞机杯呢。你哥哥下午的时候啊，就把我当成飞机杯，抱着我的肥腿使劲后入，使劲操呢。用他的大鸡巴一下一下的抽查我的骚逼，用龟头啊使劲顶我的子宫。哎呀，操的我呀白眼直翻，口水直流呢。像一条母狗一样，只会喊主人主人呢哼。😊他呀还在我的大奶子上写了厕所两大字，然后叫了几个人过来，把我当成公共厕所，让我吃他们的惊，喝他们的尿呢，我就是个飞机呗。是个鸡巴套子，不过就算是这样，也比你这种废物要好得多。小鸡巴废物，拿起飞机杯。😊想象着你老婆，我被陌生人操的欲现欲死的样子，给我使劲撸，现在开始计时。你哥的大鸡巴真的操死我了，抱着我的黑丝肥臀呢，啪啪啪的用鸡巴撞我的大屁股，把我操的双腿发软，白眼直翻。你哥哥最喜欢我的黑丝臊屁股了，他说呀，只有结婚以后的熟女，才能有这样丰腴肉感的大腿和屁股，这样操起来才最带感呢。哎呀，不过呀，我估计你这辈子是没机会享着福气了。哎，怎么了？你继续撸啊啊，你已经射了。哎呦，你可笑死老娘了，这才几秒呀，35秒还破记录了呀。😊看你这个废物东西小鸡巴废物，你这种小鸡巴就不配娶媳妇儿，我当时简直是瞎了眼，才会嫁给你这么个小鸡巴废物。不过也好，要是不和你结婚，我都不认识你哥，我见到你哥才知道。😔原来两兄弟之间的鸡巴可以差距这么大的呀，哥哥的鸡巴那么大，操的我骚逼呀，饮水直流哼。然而，弟弟的鸡巴不仅短小，而且秒射，又小又没用，居然还想着操女人，你这种废物鸡巴除了尿尿。还有点别的用吗？啊，怎么还会幻想着可以蹭女人呢？你就应该乖乖的撸你的鸡吧。毕竟没有哪个女人看到你的小鸡巴，可以忍住笑出声来。哼再说了，看我被别人操，你不是挺高兴的吗？😊真的是没有见过你这种老婆被干了，即巴还硬起来的变态呢。哦，对了，忘了告诉你了，他们呢还录了操我的录像呢。要不要看看别人的大鸡巴是怎么干我的骚逼的呢？啊，我跟你说呀，他们几个大男人轮流来操我的逼。而且都不带套，还有几个男的我都不认识，就拽了我的大屁股，把我当成飞机杯，狠狠的干我，就像干骚狗那样呢。哎呦。其他男人的鸡巴就是大呀，不知道比你这个废物早些养美男的鸡巴好到哪里去呢？真的是可笑啊。每次我扫微痒的时候，我老公都没办法满足我。没办法，我也只好去找你哥来操我了。其实啊我跟你说，对于我这种骚货来说。😔只要是个男人，是个鸡巴够大的男人就可以了。当然了，我也只对其他男人发骚。至于你这种废物东西啊，这种看着自己老婆被操还能硬起来的废物鸡巴，我碰你一下，都是对你的恩惠了。还不跪下来谢谢我，你这个废物贱狗，哎，看到没有啊，他们的鸡巴都是那种塞进来的。😊把我的骚逼慢慢撑开，然后齐根末弱，往死里操我，你都不知道他们大击吧，插进我的骚血里的时候，我是有多充实啊。心想啊，这才是做女人的快乐呀。和你睡在一起啊，就觉得浑身不舒服，觉得恶心。😊我怎么会嫁给你这么个阳痿早泄的男人呢？舔都不想给你舔，就连给你撸管儿都是满脸写着嫌弃。怎么会有你这种鸡巴那么小还秒杀的男人呢啊，你知道我被那些大鸡巴干的多爽吗？😮我觉得被那些大鸡巴操逼好幸福啊。😊肉血里面被打鸡巴塞得满满的，他们每一次冲刺都把大龟头撞在我的子宫上呢，每一次抽查都把我干的快要高潮了。哎，还有啊他们精液射进来的时候又浓又多，连续不停的在我的子宫里面射上十几下。每次他们射完，我都感觉我的子宫里面满满的都是他们的精液呢。哎，说不定我现在抠一抠，还能从骚逼里面抠出来他们的精液呢。而且呀我还沉浸在上一个人内受的快感当中的时候，下一根鸡巴都已经重新插进我的骚血里面了。我告诉他们，让我休息一下，他们都说不行呢，说我太骚了，忍不住了。哎呀，没办法，在大鸡巴男人面前呢，我就是这么骚，这么淫荡。被大鸡巴男人干的时候，我总是忍不住喊他们老公，喊他们主人，我求了他们打我的大屁股，我真是爱死这种沉浮在男人胯下的感觉了。哎，对了，他们呢可喜欢听我说骚话了，他们呢总让我喊他们爸爸哥哥来操我呀什么的，我也超级喜欢这种感觉呢。而且呀上一个人刚刚射完，他还会很懂得把软下去的鸡巴放在我的嘴里，让我给他吸鸡巴呢，吸那些还没有射出来的精液。舔他们的龟头做他们的蛋蛋，哎呦，他们的蛋蛋可大了，我觉得他们连续早一整个晚上都没有问题呢。然后几个男人就坐在边上等着，你哥坐在旁边专门录像。哎，你看你看看到这个男的大鸡巴朝小穴的镜头了吗？哇，你看没看到饮水的被操的从骚逼里面溅出来了。哎呦，看的我现在骚逼又有些痒了呢。哎喂喂喂，你那是什么表情啊？😮怎么你还指望现在可以操我呀，哼别想了，能给你看看我被其他男人操的视频，你就应该学会满足懂吗？😊自己撸你那根短小的鸡巴难道不好吗？我把视频发给你以后不要再对我说和你做爱这种话了，知道吗？实在是笑死人了，你以后啊就看着我被别人干的视频，先录一遍，求求你这个阳痿早泄男，不要再再回到家里的时候发情了啊。😊不然我晚上睡觉，一想到我旁边这个阳痿早线男在发情，我就很郁闷，觉都睡不好。以后啊你白天去上班。然后我就叫你哥带那些鸡巴蛋的男人来家里操我，然后录像录完发给你怎么样哼。是不是觉得很赚呢，又能满足自己被绿的爱好，还能够看着撸鸡吧，真的是一举两得呢，这样。回到家你也不会发情了，我也不用应付你了，简直perfect啊，什么你想要当场看。😊当场看，我怕你怀疑人生过于自卑呢。你都不知道我在那些男人面前有多贱有多骚。我 在他们面前呢完全就是一条欠操的扫母狗呢。要么高高的翘了大屁股，回头用淫荡的眼神瞪着他们后入啊，要么就是自己大大的分开自己的双腿，等着别人的插入。他们想什么时候设就什么时候设，想设在哪里呀，就设在哪里呢。有一次他们设了好多法，弄得我脸上、奶子上扫血脸，到处都是他们的精液。到最后我都不知道到底做了多久，睡了几次呢？我就记得呀，我浑身全是男人的经液。睁开眼睛就是等着我舔的鸡巴，而且他们呢还特别喜欢在我们卧室里面坐。说是要给我老公看看，没办法满足自己老婆的男人到底会经历些什么？哎呀，我看你鸡巴那么小，就算是绿帽奴。肯定也受不了那么多男人轮奸自己老婆的场面吧。你要想想，那可是好几个大鸡巴男人。把我狠狠的按在床上，一个不停的操啊，他们每一个男人的鸡巴都又粗又大又长。就好像欧美A片里面的男主角呢。就好像是一根根长长的棍子，捅 入我这个亚洲骚逼里面，把我干得死去活来的，而且还都是内设呢。到时候孩子你可得养啊，毕竟我可是你老婆，还有啊我不履行做妻子的义务，也是因为你自己阳痿早泄导致的。😊再说了，和你做怎么可能会有孩子呢？早点戒种，也不失为一件好事啊，什么。无论如何都想要看，你确定要亲眼看着自己的老婆，被几个大鸡巴男人轮奸内设吗？你确定要看我在别的男人胯下满嘴骚话，叫着别的男人老公和爸爸吗？哼？好吧好吧，既然你执意要求的话，反正你到时候自己撸社了也就没你啥事儿了。到时候看到一半的时候。可不要忽然说要证明自己什么的，我只要求你到时候不要哭出来就可以了，怪丢人的那咱们可说好了啊。我现在可要打电话约人了啊喂哥啊。你那几个兄弟今晚有没有时间啊啊，有的呀嗯那就叫上晚上一起到我这儿啊。哎呀，你管你弟干啥呀，你这管叫人来就是了。嗯，早点来啊。好嗯等你哦，哼。😊老公，你晚上可得顶住了。
。。。"""

from txt.sellcx import sell
from txt.tubercx import tuber
from txt.yyk import yyk
from txt.blk import blk




#dst=yyk
#dst=tuber
dst=sell
dst=blk


from txt.teacher0 import teacher
#dst=teacher

from txt.readfatcx import fat
#dst=fat

from txt.readshare3 import share
dst=share

from txt.dog import dog
dst=dog

from txt.lady import lady
dst=lady

from txt.egg import egg
dst=egg

from txt.touchm import touch
dst=touch

from txt.read1cx1 import read1
dst=read1

from txt.dogcx1 import dog
dst=dog

from txt.readpregcx import readpreg
dst=readpreg

from txt.forgive import forgive
dst=forgive

from txt.povntrqujv import povntrqj
dst=povntrqj

from txt.ntrslavedog import ntrslavedog
dst=ntrslavedog

from txt.ntrfoot import ntrfoot
dst=ntrfoot

from txt.av import av
dst=av

from txt.ffbutt import ffbutt
dst=ffbutt

from txt.foot import foot
dst=foot

from txt.sellroad import sellroad
dst=sellroad

from txt.green import green
dst=green

from txt.seedcx import seed
dst=seed

from txt.hooker import hooker
dst=hooker

from txt.cumclub import cumclub
dst=cumclub

from txt.wed import wed
dst=wed

from txt.begger import begger
dst=begger


txt_process='''
<br>        _
\n          _
我          你
她          我
品茹        我
妻子        我
老婆我      我
你老婆      我
你的老婆    我
老婆        我
他们        我们
他俩        我俩
肉棒        鸡巴
老二        鸡巴
肏          操
草          操
屄          逼
鲍          逼
中出        内射
保险套      避孕套
台湾        北京
阿端        __
嗯          哦
'''

#inference_instruct2
ins = """
操<cao4>
长<chang2>
腿<tui3>
稀<xi1>
痉<jing1>
"""

src = """
我是早就在考虑的，你没办法，去实践它就没法落地因为我没有时间啊~ 我为什么要今晚来烫头发呀我是有病么我快饿~死~了~ 我发了张照片儿。啊…我再想想哦。我倒是，想改善一下儿，睡眠什么的。唉…我真的是废人。我十二点的时候，去那边儿咨询了一下。
"""

"""
我怎么考虑呀？我不能现在开始找工作吧，那他要不裁我呢？就是我怎么能，在现在这阶段去找到一家下家这不可能对不对。那如果不是干这行业的话那我要做什么那这个东西，怎么规划，就是怎么落地就很麻烦啊。然后我如果还是现在这样的话正常工作的话我哪有时间去跟进这件事情呀？
"""

"""


那~可不一定~ 小看我。
并没有~ 我最近皮肤不好。

我在煮银耳汤。
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

inpathff = './asset/ff.wav'
prompt_speech_16kff = load_wav(inpathff, 16000)
srcff = "你说，他也没用过你，瞎惦记什么呀，你要搁我我都不琢磨，谁知道好不好用啊，真是的。"

inpathcc = './asset/cx.wav'
prompt_speech_16kcc = load_wav(inpathcc, 16000)
srccc = "不能穿黑丝了，黑丝有点儿冷了晚上。我，周六约，朋友出去，吃饭，你要不要一起。不然呢，我还换个衣服呀? 嘻~ 看演唱会去喽~ 得补充胶原蛋白~ 那可不一定，小看我。并没有，最近皮肤不好。就这么聊，聊的越来越好啦。"

line_num = 0
#single_line = '__'.join(dst.splitlines()) 
#single_line = smart_replace(single_line)
#single_line = single_line.replace('__', '\n')
total_lines = len(dst.splitlines())
for line in dst.splitlines():
    print(f"{line_num}/{total_lines} ... 处理行: {line}")

    if len(trim_all_wwitespace(line)) < 1:
        line_num += 1
        continue

    if 'ww' in line:
        line = line.replace('@@ww', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcww, prompt_speech_16k=prompt_speech_16kww, stream=False, speed=1)):
            filename = 'cx_{:03d}_{:03d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif 'dd' in line:
        line = line.replace('@@dd', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcdd, prompt_speech_16k=prompt_speech_16kdd, stream=False, speed=1)):
            filename = 'cx_{:03d}_{:03d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif 'ss' in line:
        line = line.replace('@@ss', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcss, prompt_speech_16k=prompt_speech_16kss, stream=False, speed=1)):
            filename = 'cx_{:03d}_{:03d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif 'ff' in line:
        line = line.replace('@@ff', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srcff, prompt_speech_16k=prompt_speech_16kff, stream=False, speed=1)):
            filename = 'cx_{:03d}_{:03d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    elif 'cc' in line:
        line = line.replace('@@cc', '')
        line = line.replace('@@', '')
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=srccc, prompt_speech_16k=prompt_speech_16kcc, stream=False, speed=1)):
            filename = 'cx_{:03d}_{:03d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)
    else:
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text=line, prompt_text=src, prompt_speech_16k=prompt_speech_16k, stream=False, speed=1)):
            filename = 'cx_{:03d}_{:03d}.wav'
            torchaudio.save(filename.format(line_num, i), j['tts_speech'], cosyvoice.sample_rate)


    line_num += 1




output_list = find_cx_wav_files()
output_filename = 'cx.wav'
print('[cx] output_list ...', output_list)

concatenate_wavs_pydub(output_list, output_filename)

delete_files(output_list)

print('[cx] output_filename ...', output_filename)

print('[cx] Done.')


