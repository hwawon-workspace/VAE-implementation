# 데이터셋_모델_배치사이즈_에포크_학습률로 이미지 저장할 파일명 만들기

def generate_filename(config):
    dataset_name = config['data']['dataset']
    model_class_name = config['model']['model']
    batch_size = config['data']['batch_size']
    epochs = config['model']['epochs']
    learning_rate = config['model']['learning_rate']

    # 파일 이름 구성
    filename = f"{dataset_name}_{model_class_name}_bs{batch_size}_ep{epochs}_lr{learning_rate}"
    return filename