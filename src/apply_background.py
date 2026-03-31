import argparse
from PIL import Image

def blend_text_with_background(text_image_path, bg_image_path, output_path):
    # 1. Բեռնում ենք մեր գեներացված թափանցիկ (RGBA) տեքստը
    try:
        text_img = Image.open(text_image_path).convert("RGBA")
    except Exception as e:
        print(f" Սխալ: Չի գտնվել տեքստի նկարը: {e}")
        return

    # 2. Բեռնում ենք ֆոնի նկարը (մագաղաթ կամ տետրի թուղթ)
    try:
        bg_img = Image.open(bg_image_path).convert("RGBA")
    except Exception as e:
        print(f" Սխալ: Չի գտնվել ֆոնի նկարը: {e}")
        return

    # 3. Քանի որ գեներացված տեքստը կարող է շատ երկար լինել (օրինակ՝ ամբողջ նախադասություն),
    # Մենք պետք է ֆոնի նկարի չափսերը հարմարեցնենք տեքստի չափսերին:
    text_width, text_height = text_img.size

    # Կտրում ենք ֆոնի կենտրոնական մասը տեքստի չափով (Center Crop)
    bg_width, bg_height = bg_img.size

    # Եթե ֆոնը փոքր է, մեծացնում ենք (Resize) այն պահպանելով համամասնությունը
    scale = max(text_width / bg_width, text_height / bg_height)
    new_bg_width = int(bg_width * scale)
    new_bg_height = int(bg_height * scale)

    bg_img = bg_img.resize((new_bg_width, new_bg_height), Image.LANCZOS)

    # Կենտրոնական կտրում (Crop)
    left = (new_bg_width - text_width) / 2
    top = (new_bg_height - text_height) / 2
    right = (new_bg_width + text_width) / 2
    bottom = (new_bg_height + text_height) / 2

    bg_canvas = bg_img.crop((left, top, right, bottom))

    # 4. Ռեալիստական Ձուլում (Alpha Compositing)
    # Սա ուղղակի չի դնում նկարը վրեն, այլ թանաքը "ներծծում" է թղթի մեջ
    final_image = Image.alpha_composite(bg_canvas, text_img)

    # Վերադարձնում ենք RGB (առանց թափանցիկության) և պահպանում
    final_image = final_image.convert("RGB")
    final_image.save(output_path, quality=100)
    print(f" Հիանալի է! Ֆոնով նկարը պահպանվեց որպես '{output_path}'")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Դնում է թափանցիկ տեքստը ֆոնի վրա')
    p.add_argument('--text_img', type=str, required=True, help='generate_3.py-ի տված նկարը (output.png)')
    p.add_argument('--bg_img', type=str, required=True, help='Ֆոնի նկարը (օրինակ paper.jpg)')
    p.add_argument('--output', type=str, default='final_realistic.jpg', help='Պատրաստի նկարի անունը')

    args = p.parse_args()
    blend_text_with_background(args.text_img, args.bg_img, args.output)