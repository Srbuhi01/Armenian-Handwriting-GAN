import argparse
import numpy as np
from PIL import Image, ImageOps


def calculate_brightness(image_region):
    """
    Հաշվում է նկարի տրված հատվածի միջին պայծառությունը (Luminance):
    Օգտագործվում է ստանդարտ հեռուստատեսային բանաձևը (BT.601):
    Y = 0.299*R + 0.587*G + 0.114*B
    Վերադարձնում է արժեք 0-ից (սև) մինչև 255 (սպիտակ):
    """
    # Վերածում ենք մատրիցի (Numpy Array)
    arr = np.array(image_region.convert("RGB"))

    # Կիրառում ենք ընկալման (Perceptual) կշիռները
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Վերադարձնում ենք միջին արժեքը
    return np.mean(luminance)


def auto_color_overlay(text_image_path, bg_image_path, output_path):
    # 1. Բեռնում ենք նկարները
    try:
        text_img = Image.open(text_image_path).convert("RGBA")
        bg_img = Image.open(bg_image_path).convert("RGBA")
    except Exception as e:
        print(f"❌ Սխալ ֆայլերը կարդալիս: {e}")
        return

    # 2. Մասշտաբավորում ենք տեքստը (որպեսզի զբաղեցնի ֆոնի լայնության 70%-ը)
    target_width = int(bg_img.width * 0.7)
    scale = target_width / text_img.width
    target_height = int(text_img.height * scale)

    # LANCZOS ֆիլտրը ապահովում է բարձր որակ փոքրացնելիս/մեծացնելիս
    text_img = text_img.resize((target_width, target_height), Image.LANCZOS)

    # 3. Որոշում ենք Կոորդինատները (Վերևի Կենտրոն)
    x_pos = (bg_img.width - text_img.width) // 2
    y_pos = int(bg_img.height * 0.05)  # Վերևից 5% հեռավորություն

    # 4. ԿՏՐՈՒՄ ԵՆՔ ՖՈՆԻ ԱՅՆ ՀԱՏՎԱԾԸ, ՈՐՏԵՂ ԼԻՆԵԼՈՒ Է ՏԵՔՍՏԸ (ROI)
    # Սա կրիտիկական է. մենք հաշվում ենք ոչ թե ամբողջ նկարի պայծառությունը, այլ միայն տեքստի տակի ֆոնի:
    roi_box = (x_pos, y_pos, x_pos + text_img.width, y_pos + text_img.height)
    bg_roi = bg_img.crop(roi_box)

    # 5. ՀԱՇՎՈՒՄ ԵՆՔ ՊԱՅԾԱՌՈՒԹՅՈՒՆԸ ԵՎ ԱՎՏՈՄԱՏ ՈՐՈՇՈՒՄ ԳՈՒՅՆԸ
    avg_brightness = calculate_brightness(bg_roi)
    print(f"📊 Ֆոնի տվյալ հատվածի միջին պայծառությունը. {avg_brightness:.2f} (0=Սև, 255=Սպիտակ)")

    # Կոնտրաստի շեմ (Threshold)
    threshold = 127

    # Քանի որ մեր օրիգինալ տեքստը ՍԵՎ է (0,0,0, Alpha),
    # եթե ֆոնը մուգ է (պայծառությունը ցածր է շեմից), մենք տեքստը դարձնում ենք ՍՊԻՏԱԿ:
    if avg_brightness < threshold:
        print("🌙 Ֆոնը ՄՈՒԳ է: Ալգորիթմը ավտոմատ ընտրեց ՍՊԻՏԱԿ տեքստ (RGB Inversion):")

        # Անջատում ենք ալիքները (Red, Green, Blue, Alpha)
        r, g, b, a = text_img.split()

        # Միացնում ենք միայն գույները և շրջում (Invert՝ սևը դառնում է սպիտակ)
        rgb_img = Image.merge('RGB', (r, g, b))
        inverted_rgb = ImageOps.invert(rgb_img)

        # Վերամիավորում ենք սպիտակ գույնը օրիգինալ թափանցիկության (Alpha) հետ
        text_img = Image.merge('RGBA', (*inverted_rgb.split(), a))
    else:
        print("☀️ Ֆոնը ԼՈՒՍԱՎՈՐ է: Ալգորիթմը պահպանեց ՍԵՎ տեքստը:")

    # 6. Վերադրում (Alpha Compositing)
    # Երրորդ արգումենտը `text_img` ծառայում է որպես դիմակ (mask), որպեսզի միայն տառերը երևան
    bg_img.paste(text_img, (x_pos, y_pos), text_img)

    # 7. Պահպանում
    final_image = bg_img.convert("RGB")
    final_image.save(output_path, quality=100)
    print(f"✅ Պատրաստ է! Խելացի կոնտրաստով վերջնական նկարը պահպանվեց որպես '{output_path}'")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Ավտոմատ որոշում է տեքստի գույնը ըստ ֆոնի պայծառության')
    p.add_argument('--text_img', type=str, required=True, help='generate_3.py-ի տված նկարը (օր. output.png)')
    p.add_argument('--bg_img', type=str, required=True, help='Ֆոնի նկարը (օր. mountains.jpg)')
    p.add_argument('--output', type=str, default='smart_overlay_result.jpg', help='Արդյունքի ֆայլի անունը')

    args = p.parse_args()
    auto_color_overlay(args.text_img, args.bg_img, args.output)