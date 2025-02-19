from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from config import SQUARE_SIZE,dpi, width_inch, height_inch, supported_extensions
import fitz

def crop_center_square(image, size=SQUARE_SIZE):
    """
    Вырезание квадрата выше центра из картинки.
    """
    width, height = image.size

    left = (width - size) // 2
    top = (height - size) // 5
    right = left + size
    bottom = top + size

    # Вырезаем квадрат из центра (чуть выше центра)
    cropped_image = image.crop((left, top, right, bottom))

    # Сохраняем обрезанное изображение
    return cropped_image

class DocumentDataset(Dataset): #класс для модификации датасета плд вход нейронки
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['normal', 'flipped']
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = crop_center_square(image)
        image = self.transform(image)

        return image, label

TRANSFORM = transforms.Compose([
    transforms.Resize((SQUARE_SIZE, SQUARE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #стандартные значения для RESNET
])


def change_dpi_A4(image_path, result_name_with_path, dpi=dpi, width_inch=width_inch, height_inch=height_inch):
    """
    Смена dpi картинки на основе алгоритма маштабирования через бикубическую интерполяцию.
    Считаем что на входе идет лист А4.
    :param image_path: Путь до фото
    :param result_name_with_path: Новоое назвние и возможный путь результата.
    :param dpi: Нужный dpi.
    :param width_inch: Ширина в дюймах.
    :param height_inch: Длина в дюймах.
    """
    desired_width_inch = width_inch
    desired_height_inch = height_inch
    new_dpi = dpi
    new_width = int(desired_width_inch * new_dpi)
    new_height = int(desired_height_inch * new_dpi)
    with Image.open(image_path) as img:
        new_img = img.resize((new_width, new_height), Image.BICUBIC)
        new_img.save(result_name_with_path, dpi=(new_dpi, new_dpi))



def rotate_one(image_path, result_name_with_path):
    """
    Переворачивает фото на 180 градусов
    :param image_path: Путь до фото.
    :param result_name_with_path: Новоое назвние и возможный путь результата.
    """
    with Image.open(image_path) as img:
        rotated_img = img.rotate(180)
        rotated_img.save(result_name_with_path)


def rotate_mult(folder_path, output_path):
    """
    Переворачивает все фото в папке.
    :param folder_path: Путь до папки.
    :param output_path: Путь до папки для результата.
    """
    for filename in os.listdir(folder_path):
        # Проверяем расширение файла
        if filename.lower().endswith(supported_extensions):
            # Полный путь к файлу
            file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(output_path, filename)
            try:
                # Открываем изображение
                with Image.open(file_path) as img:
                    # Переворачиваем изображение на 180 градусов
                    rotated_img = img.rotate(180)
                    # Сохраняем перезаписывая исходный файл
                    rotated_img.save(output_file_path)
                    # print(f"Изображение перевернуто: {filename}")
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {str(e)}")
    print("Обработка завершена!")
def pdf_to_images(pdf_path, output_folder, image_format="jpg", zoom=2):
    """
    Преобразует PDF-файл в набор изображений.

    :param pdf_path: Путь к PDF-файлу.
    :param output_folder: Папка для сохранения изображений.
    :param image_format: Формат изображений.
    :param zoom: Масштаб для увеличения качества изображения.
    """
    # Открываем PDF-файл
    doc = fitz.open(pdf_path)

    # Создаем папку для сохранения изображений, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Проходим по каждой странице PDF
    for page_num in range(len(doc)):
        # Загружаем страницу
        page = doc.load_page(page_num)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        # Увеличиваем масштаб для улучшения качества изображения
        mat = fitz.Matrix(zoom, zoom)

        # Преобразуем страницу в изображение (пиксмап)
        pix = page.get_pixmap(matrix=mat)

        # Сохраняем изображение
        image_path = os.path.join(output_folder, f"{pdf_name}_page_{page_num + 1}.{image_format}")
        pix.save(image_path)

        #print(f"Страница {page_num + 1} сохранена как {image_path}")

    #print(f"Все страницы сохранены в папке {output_folder}")
def process_multiple_pdfs(pdf_paths, output_folder, image_format="jpg", zoom=2):
    """
    Обрабатывает несколько PDF-файлов.

    :param pdf_paths: Список путей к PDF-файлам или путь к папке с PDF-файлами.
    :param output_folder: Папка для сохранения изображений.
    :param image_format: Формат изображений.
    :param zoom: Масштаб для увеличения качества изображения.
    """
    # Если передан путь к папке, получаем все PDF-файлы в этой папке
    if isinstance(pdf_paths, str) and os.path.isdir(pdf_paths):
        pdf_folder = pdf_paths
        pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    # Обрабатываем каждый PDF-файл
    for pdf_path in pdf_paths:
        if not os.path.isfile(pdf_path):
            print(f"Файл {pdf_path} не найден, пропускаем.")
            continue

        print(f"Обработка файла: {pdf_path}")
        pdf_to_images(pdf_path, output_folder, image_format, zoom)

def rename_files_in_folder(folder_path, name_template="file_{num}", start_num=1, prefix="", suffix=""):
    """
    Переименовывает файлы в указанной папке.

    :param folder_path: Путь к папке с файлами.
    :param name_template: Шаблон для нового имени файла.
    :param start_num: Начальный номер для нумерации файлов.
    :param prefix: Префикс, который будет добавлен к имени файла.
    :param suffix: Суффикс, который будет добавлен к имени файла.
    """
    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует.")
        return

    # Получаем список файлов в папке
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # Сортируем файлы по имени

    # Переименовываем файлы
    for i, filename in enumerate(files, start=start_num):
        # Получаем расширение файла
        file_ext = os.path.splitext(filename)[1]

        # Формируем новое имя файла
        new_name = name_template.format(num=i)
        new_name = f"{prefix}{new_name}{suffix}{file_ext}"

        # Полные пути к старому и новому имени
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)

        # Переименовываем файл
        os.rename(old_file, new_file)
        #print(f"Переименован: {filename} -> {new_name}")

    print(f"Все файлы в папке {folder_path} переименованы.")