# 이미지를 읽는 함수 정의
def get_image(p: str):
    return Image.open(p).convert("L")

# 데이터셋 클래스 정의
class baseDataset(Dataset):
    def __init__(self, root: str, train: bool):
        super().__init__()

        # 훈련 데이터셋 또는 검증 데이터셋의 루트 디렉토리 설정
        if train:
            self.root = os.path.join(root, "train")
            self.transform = T.Compose([
                T.ToTensor()
            ])
        else:
            self.root = os.path.join(root, "val")
            self.transform = T.Compose([
                T.ToTensor()
            ])

        # 데이터 리스트 생성
        data_list = []
        for i in range(10):
            dir = os.path.join(self.root, str(i))
            for img in os.listdir(dir):
                img_path = os.path.join(dir, img)
                data_list.append((i, img_path))
        self.data_list = data_list

    def __len__(self):
        # 데이터셋의 총 데이터 개수 반환
        return len(self.data_list)

    def __getitem__(self, idx: int):
        # 인덱스에 해당하는 데이터 반환
        number, img_path = self.data_list[idx]

        # 이미지 파일을 PIL 객체로 읽어들이고, 그레이스케일로 변환한 후 텐서로 변환
        img_obj = get_image(img_path)
        img_tensor = self.transform(img_obj)

        return img_tensor, number



# 데이터셋 정의
#train_dataset = baseDataset(MNIST_ROOT, True)
train_dataset = datasets.MNIST(
    root      = './.data/',
    train     = True,
    download  = True,
    transform = transforms.ToTensor()
)
print("Train dataset의 개수",len(train_dataset))

#val_dataset = baseDataset(MNIST_ROOT, False)
val_dataset = datasets.MNIST(
    root      = './.data/',
    train     = False,
    download  = True,
    transform = transforms.ToTensor()
)
print("Validation dataset의 개수",len(val_dataset))

# 데이터로더 정의
train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, True)

# 원본 이미지를 시각화 하기 (첫번째 열)
view_data = train_dataset.data[:5].view(-1, 28*28) #view?
view_data = view_data.type(torch.FloatTensor)/255. #255?