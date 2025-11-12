import markovify
import re
import random
import markovify
import os
from augraphy import DirtyDrum, SubtleNoise, Scribbles, LowInkRandomLines, InkBleed, AugraphyPipeline
import cv2
from PIL import Image, ImageDraw, ImageFont
text = "hɔi xʷiý ä'säl täto äli'o kuḿ 'qät'p xʷiý äli'c ä'säl. tc:taq́ä't'p ha sqäĺsi' (wɫku̥mc) hä ttćä'säl tmu's:lc äkʷn täíxʷi 'itsxu'i äkʷn t'ä'xʷu̥ntmät stsi'gwtu̥mc hɔi t'i' tsi'ˡ hahui tcńadosaĺqsi'ẃäs äku'stɛm ku'ᵘtci''t nᵃ\u0308kunk'ʷä'tpɛntsut xʷiýa sqäĺsi' xʷä 'äku'stɛm t'i'ˡ ɫä kʷints ɫä ẃ:ĺẃ:ĺi'ḿs  xʷi'ˡɫ h:nk'ʷät'pɛntsut xʷa sqäísi' tciya'ĺ'ĺxʷqɛn k'ʷäý atciya'ĺxʷqɛn ɫa 'äɫ ɛɫä'xʷä'äs lä'djɛntɛm. xä'ʀɛntsut äku'stus ɫa si'nstä'äs näᵃ\u0308 kux̥al kutcsxʷu̥t'i'ɫtśä'ᵃ\u0308 tsäaᵃ\u0308 ttcnä'k'ʷä'ᵃ\u0308 xʷi'ˡɫ tśäli'c:c tciya'ĺxʷqɛn xʷi'ˡ tiýt́i'ýäqʷt utśa'tśɛx̥ms xʷiýä q́ʷi'ˡɫ ':tsq́ʷä'ᵃ\u0308ɫälwi's  xu''l pänä'ᵃ\u0308 suuyäpu̥mc 'astq́ʷi' tsi'ˡɫ c:ɫ t'ut'u̥k'ʷi'ĺup. xʷiýa suᵘyä' umäĺtsi'': 'äᵃ\u0308mi't'tćädä'ᵃ\u0308 tcäɫkusx̥ɛnäʙɛngwh'ýqɛn lut(h)äᵃ\u0308ýi'ɫɛnts xuxʷiýä guɫqaqi'tstcs i'ɫn:lc ɫ gwä'ýtsɛn i't'ɛt.c. ätśɛmu'ɫn:lc ɫa sts:tsa'saĺqs. xʷist ɫuẃä tätc (h)nlä'q́ɛntsutɛn ńuɫxʷ q́äsp sɛnkʷi''ts ttsaqi'nä'ämɛnts itsgwä'lp ɫuẃä tsä'txʷs:lc tä'aᵃ\u0308tcu'sɛm xʷa sts:tsa'saĺqs hiɫ gwi'tcts itsgwä'lp tsätxʷs:lc pu'ᵘpu'ᵘsi'ntc cɛtćmi'nts ut'uqʷa''' sci''tɛmc k'uk'ʷuńi'ýä'ᵃ\u0308 uɫmiĺtsi'ˡˠ ut'ugʷä''s stsɛtai'ẃtɛḿc täĺtsi'ˡ uupɛ̤'s quqʷa'ᵃqʷä'äĺ pɛpu'ĺutätcuĺ akʷn nä'ᵃ\u0308tätcnlä'q́ɛntsutɛn nxatxati'ɫtśä'äntp hnṕɛṕa'q́ʷɛńts'asx̥ux̥ʷɔḿqɛn ni''tɛpä'x̥‿äẃäsɛnts yu'tśäm:s sxu'ᵘxu'ᵘni'toäɫp "


chars_to_remove = ['.', ',', '!', '?', '[', ']', '<', '>', '/', '\\']
cleaned_text = text.lower()
for char in chars_to_remove:
    cleaned_text = cleaned_text.replace(char, ' ')
cleaned_text = re.sub(r'[\n\t\r\s]+', ' ', cleaned_text).strip()
word_list = [word.strip() for word in cleaned_text.split(' ') if word.strip()]
corpus_for_char_model = "\n".join(word_list)

class CharText(markovify.NewlineText):
    def word_split(self, sentence):
        return list(sentence)
    def word_join(self, words):
        return "".join(words)
gen_words = []
text_model = CharText(corpus_for_char_model, state_size=2) 
for i in range(2000):
    new_word = text_model.make_sentence(
        tries=50,
        max_words=10,
        min_words=2
    )
    gen_words.append(new_word)
def generate_sentence(words, min_words=4, max_words=8):
    n_words = random.randint(min_words, max_words)
    sentence = []
    for _ in range(n_words):
        sentence.append(random.choice(words))
    return " ".join(sentence) + "."

def generate_image(sentence):
    img = Image.new("RGB", (1800, 200), color="white")
    draw = ImageDraw.Draw(img)
    #font_path = "sources/fonts/ITC Stone Serif Phonetic IPA Regular.ttf"
    font_path = "sources/fonts/Charis-Medium.ttf"
    font_size = 48
    font = ImageFont.truetype(font_path, font_size)
    draw.text((100,100), text=sentence, font=font, fill="black")
    return img
ink_phase = [
        LowInkRandomLines(count_range=(5, 10),
                                                    use_consistent_lines=True,
                                                    noise_probability=0.1,
                                                    ),
        InkBleed(intensity_range=(0.4, 0.7),
                    kernel_size=(5, 5),
                    severity=(0.3, 0.5)
                    )
]
post_phase = [
    DirtyDrum(line_width_range=(1, 4),
                      line_concentration=0.1,
                      direction=random.choice([0,1]),
                      noise_intensity=random.uniform(.3,.5),
                      noise_value=(0, 30),
                      ksize=(3, 3),
                      sigmaX=0,
                      ),
    SubtleNoise(subtle_range=25),
    Scribbles(
                scribbles_type="random",
                scribbles_location="random",
                scribbles_ink = "pen",
                scribbles_size_range=(100, 200),
                scribbles_count_range=(1,2),
                scribbles_thickness_range=(1, 2),
                scribbles_brightness_change=[32, 64, 128],
                scribbles_text="random",
                scribbles_text_font="random",
                scribbles_text_rotate_range=(0, 360),
                scribbles_lines_stroke_count_range=(1, 3),
                scribbles_color=(27,27,27),
                p = .3
            ),

    
]
pipeline = AugraphyPipeline(ink_phase=ink_phase, post_phase=post_phase)
os.makedirs("./mark_synthetic", exist_ok=True)
"""
    
for i in range(500):
    sentence = generate_sentence(gen_words)
    image = generate_image(sentence)
    path = f"./mark_synthetic/sample_{i}.png"
    image.save(path)
    im = cv2.imread(path)
    augmented_image = pipeline(im)
    cv2.imwrite(path, augmented_image)
    with open(f"./mark_synthetic/sample_{i}.txt", "w", encoding="utf-8") as f:
        f.write(sentence)
    """

os.makedirs("./cda", exist_ok=True)
os.makedirs("./cda/train", exist_ok=True)
os.makedirs("./cda/val", exist_ok=True)
for i in range(500):
    sentence = generate_sentence(gen_words)
    image = generate_image(sentence)
    path = f"./cda/train/sample_{i}.png"
    image.save(path)
    im = cv2.imread(path)
    augmented_image = pipeline(im)
    cv2.imwrite(path, augmented_image)
    with open(f"./cda/train/sample_{i}.txt", "w", encoding="utf-8") as f:
        f.write(sentence)
for i in range(100):
    sentence = generate_sentence(gen_words)
    image = generate_image(sentence)
    path = f"./cda/val/sample_{i}.png"
    image.save(path)
    im = cv2.imread(path)
    augmented_image = pipeline(im)
    cv2.imwrite(path, augmented_image)
    with open(f"./cda/val/sample_{i}.txt", "w", encoding="utf-8") as f:
        f.write(sentence)
    
