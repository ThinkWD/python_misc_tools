import os

import cv2
from tqdm import tqdm

from module import find_img

fonts = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
]


def put_number(img, h, w, target_size, font, thickness, text, save_path):
    # Compute base text size at scale=1
    ((_, base_h), _) = cv2.getTextSize(text, font, 1.0, thickness=1)
    # Compute scale so text height ~ target_size
    scale = target_size / base_h
    # Get actual scaled text height
    ((_, text_h_scaled), _) = cv2.getTextSize(text, font, scale, thickness=1)
    # Determine maximum reasonable thickness (e.g., 10% of text height)
    max_thick = max(1, int(text_h_scaled * 0.1))
    # Cap the requested thickness
    actual_thick = min(thickness, max_thick)
    # Recompute text size with actual scale and thickness
    ((text_w, text_h), _) = cv2.getTextSize(text, font, scale, thickness=actual_thick)
    # Position: center the text
    x = int((w - text_w) / 2)
    y = int((h + text_h) / 2)
    # Draw text on a copy to avoid overlap
    img_drawn = img.copy()
    cv2.putText(
        img_drawn, text, (x, y), font, scale, color=(255, 255, 255), thickness=actual_thick, lineType=cv2.LINE_AA
    )
    # Save output
    cv2.imwrite(save_path, img_drawn)


def process(root_path):
    print('\n[info] start task...')

    raw_path = os.path.join(root_path, 'imgs')
    assert os.path.isdir(raw_path)
    dst_path = os.path.join(root_path, 'dataset_cls')
    for number in range(0, 10):
        for font_id, font in enumerate(fonts):
            os.makedirs(f'{dst_path}/{number}/font{font_id}')

    img_list = find_img(raw_path)
    for num, file in enumerate(tqdm(img_list, leave=True, ncols=100, colour='CYAN')):
        # misc path
        raw_name, extension = os.path.splitext(file)
        raw_img = f'{raw_path}/{raw_name}{extension}'

        # Load image
        img = cv2.imread(raw_img)
        if img is None:
            print(f'[warning] failed to load image: {raw_img}')
            continue

        h, w = img.shape[:2]
        rate = h / w
        if rate < 0.34 or rate > 3:
            continue

        # Desired text size: half of image height or one-third of width, whichever is smaller
        target_size = [min(h * 0.5, w * 0.66), min(h * 0.8, w * 0.9)]

        for font_id, font in enumerate(fonts):
            for number in range(0, 10):
                text = f'{number}'
                for size in target_size:
                    # Compute base text height at scale=1
                    ((_, base_h), _) = cv2.getTextSize(text, font, 1.0, thickness=1)
                    scale = size / base_h
                    ((_, text_h_scaled), _) = cv2.getTextSize(text, font, scale, thickness=1)
                    # Determine max thickness for this text
                    max_thick = max(1, int(text_h_scaled * 0.1))
                    for thickness in range(1, min(5, max_thick) + 1):
                        save_path = f'{dst_path}/{number}/font{font_id}/{raw_name}_{size}_{thickness}.jpg'
                        put_number(img, h, w, size, font, thickness, text, save_path)


if __name__ == '__main__':
    process('D:\\User\\Desktop\\nuedc-2025c\\nudec\\imgs_crop\\imgs')  # os.getcwd()
    print('\nAll process success\n')
