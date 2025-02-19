import torch
SQUARE_SIZE = 224 #размер входной картинки в нейронку
BATCH_SIZE = 8 # размер пакета (батча)
EPOCHS = 10 # количество эпох
LEARNING_RATE = 0.001 #скорость обучения
PATIENCE = 3 #ожидание, для защиты от переобучения

dpi=100
width_inch=8.27 #формат А4 в дюймах
height_inch=11.69
supported_extensions = ('.jpg', '.jpeg') # разрешенные расширения фото

DEVISE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # где будет происходить вычисление - либо на процессоре,
                                                                      # либо на видеокарте от nvidia c cuda ядрами