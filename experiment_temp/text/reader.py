# -*- coding: utf-8 -*-
import os
from detection import get_detector, get_textbox
from recognition import get_recognizer, get_text
from utils import group_text_box, get_image_list, get_paragraph,\
                   diff, reformat_input,\
                   make_rotated_img_list, set_result_with_confidence,\
                   reformat_input_batched

from bidi.algorithm import get_display

import torch


class Code1OCR(object):
    def __init__(self, recog_network='generation2', quantize=True, cudnn_benchmark=False):

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # recognition model
        separator_list = {}
        dict_list = {}
        dict_list['ko'] = os.path.join("char/ko.txt")
        dict_list['en'] = os.path.join("char/en.txt")
        detector_path = 'weights/craft_mlt_25k.pth'
        model_path = 'weights/korean_g2.pth'
        self.model_lang = 'korean'

        # detection
        self.detector = get_detector(detector_path, self.device, quantize, cudnn_benchmark=cudnn_benchmark)

        #recognition
        if recog_network == 'generation1':
            network_params = {
                'input_channel': 1,
                'output_channel': 512,
                'hidden_size': 512
            }
        elif recog_network == 'generation2':
            network_params = {
                'input_channel': 1,
                'output_channel': 256,
                'hidden_size': 256
            }

        self.character = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘"
        self.symbol = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"
        self.lang_char = []
        char_file = ['char/ko_char.txt', 'char/en_char.txt']
        for c in char_file:
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                char_list = input_file.read().splitlines()
            self.lang_char += char_list
        self.lang_char += char_list
        self.lang_char = set(self.lang_char).union(set(self.symbol))
        self.lang_char = ''.join(self.lang_char)

        self.recognizer, self.converter = get_recognizer(recog_network, network_params, \
                                                         self.character, separator_list, \
                                                         dict_list, model_path, device=self.device, quantize=quantize)

    def detect(self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None):

        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box_list = get_textbox(self.detector, img, canvas_size, mag_ratio,
                                    text_threshold, link_threshold, low_text,
                                    False, self.device, optimal_num_chars)

        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,\
                  decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                  workers = 0, allowlist = None, blocklist = None, detail = 1,\
                  rotation_info = None,paragraph = False,\
                  contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                  y_ths = 0.5, x_ths = 1.0, reformat=True, output_format='standard'):

        imgH = 64

        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        if allowlist:
            ignore_char = ''.join(set(self.character)-set(allowlist))
        elif blocklist:
            ignore_char = ''.join(set(blocklist))
        else:
            ignore_char = ''.join(set(self.character)-set(self.lang_char))

        if self.model_lang in ['chinese_tra','chinese_sim']: decoder = 'greedy'

        if (horizontal_list==None) and (free_list==None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        # without gpu/parallelization, it is faster to process image one by one
        if ((batch_size == 1) or (self.device == 'cpu')) and not rotation_info:
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              workers, self.device)
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              workers, self.device)
                result += result0
        # default mode will try to process multiple boxes at the same time
        else:
            image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)
            image_len = len(image_list)
            if rotation_info and image_list:
                image_list = make_rotated_img_list(rotation_info, image_list)
                max_width = max(max_width, imgH)

            result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                          ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                          workers, self.device)

            if rotation_info and (horizontal_list+free_list):
                result = set_result_with_confidence(result, image_len)

        if self.model_lang == 'arabic':
            direction_mode = 'rtl'
            result = [list(item) for item in result]
            for item in result:
                item[1] = get_display(item[1])
        else:
            direction_mode = 'ltr'

        if paragraph:
            result = get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode = direction_mode)

        if detail == 0:
            return [item[1] for item in result]
        elif output_format == 'dict':
            return [ {'boxes':item[0],'text':item[1],'confident':item[2]} for item in result]
        else:
            return result

    def ocr(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img, min_size, text_threshold,\
                                                 low_text, link_threshold,\
                                                 canvas_size, mag_ratio,\
                                                 slope_ths, ycenter_ths,\
                                                 height_ths,width_ths,\
                                                 add_margin, False)
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(img_cv_grey, horizontal_list, free_list,\
                                decoder, beamWidth, batch_size,\
                                workers, allowlist, blocklist, detail, rotation_info,\
                                paragraph, contrast_ths, adjust_contrast,\
                                filter_ths, y_ths, x_ths, False, output_format)

        return result

    def readtext_batched(self, image, n_width=None, n_height=None,\
                         decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                         workers = 0, allowlist = None, blocklist = None, detail = 1,\
                         rotation_info = None, paragraph = False, min_size = 20,\
                         contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                         text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                         canvas_size = 2560, mag_ratio = 1.,\
                         slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                         width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        When sending a list of images, they all must of the same size,
        the following parameters will automatically resize if they are not None
        n_width: int, new width
        n_height: int, new height
        '''
        img, img_cv_grey = reformat_input_batched(image, n_width, n_height)

        horizontal_list_agg, free_list_agg = self.detect(img, min_size, text_threshold,\
                                                         low_text, link_threshold,\
                                                         canvas_size, mag_ratio,\
                                                         slope_ths, ycenter_ths,\
                                                         height_ths, width_ths,\
                                                         add_margin, False)
        result_agg = []
        # put img_cv_grey in a list if its a single img
        img_cv_grey = [img_cv_grey] if len(img_cv_grey.shape) == 2 else img_cv_grey
        for grey_img, horizontal_list, free_list in zip(img_cv_grey, horizontal_list_agg, free_list_agg):
            result_agg.append(self.recognize(grey_img, horizontal_list, free_list,\
                                            decoder, beamWidth, batch_size,\
                                            workers, allowlist, blocklist, detail, rotation_info,\
                                            paragraph, contrast_ths, adjust_contrast,\
                                            filter_ths, y_ths, x_ths, False, output_format))

        return result_agg


if __name__ == '__main__':
    img = 'C:/Users/home/Desktop/work/test_img/driver/A-IMG_0045.JPG'
    code1 = Code1OCR()
    result = code1.ocr(img)

    for r in result:
        box, text, conf = r
        print(text)
