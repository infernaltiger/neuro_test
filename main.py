from data_refactor import DocumentDataset, DataLoader, TRANSFORM
from model_main import DocumentOrientationModel,Main_Model_Functions
from config import BATCH_SIZE
def test_model(model_func, path):
    result = model_func.predict(path)
    print(result[2])
    return result
if __name__ == "__main__":
    model = DocumentOrientationModel()
    model_func = Main_Model_Functions(model)

    while True:
        print("Выберете вариант работы:")
        print("0 - выход")
        print("1 - обучить модель")
        print("2 - загрузить модель")
        print("3 - проверить модель на 1 фотографии")
        print("4 - проверить модель на тестирующей выборке")
        state = int(input())
        match state:
            case 0:
                break
            case 1:
                print("Обучение в тестовой версии не предусмотрено, но функции для обучения присутсвтуют")
            case 2:
                print("Введите путь к файлу с моделью (например central_square_10_epoch.pth как уже обученная модель):")
                path = input()
                model_func.load_model(path)
            case 3:
                print("Введите путь к файлу с фото документа А4 в формате jpg с dpi 100 (например файлы из папки test_files):")
                path = input()
                print(test_model(model_func,path))
            case 4:
                print("Введите путь к директории с тестовой выборкой (например testing/):")
                testing_dataset_dir = input()
                testing_dataset = DocumentDataset(testing_dataset_dir, transform=TRANSFORM)
                testing_loader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False)
                print(model_func.test_model(testing_loader))
            case _:
                print("Повторите ввод")