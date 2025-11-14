import torch, time, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union
from PIL import Image
from torchmetrics.functional import confusion_matrix
from dataclasses import dataclass
from torch import autocast, GradScaler
from typing import Optional

plt.switch_backend('agg')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def check_path(path):
	path = Path(path)
	path.mkdir(parents=True, exist_ok=True)


#TODO: assert 예외처리로 변경할 것
class TrainConfig():
	def __init__(self, save_point=30, batch_size=64, workers=12, epochs=10000, patience=10, lr=0.0005, inplace=(224,224),
				transforms:dict|None = None, criterion=nn.CrossEntropyLoss(), optimizer:optim.Optimizer|None = None):
		for param, name in zip((save_point, batch_size, workers, epochs, patience),('save_point', 'batch', 'workers', 'epochs', 'patience')):
			assert isinstance(param, int), f'{name} must be instance of int'
		assert isinstance(lr, (float, int)), 'lr must be instance of float or int'
		assert isinstance(inplace, (int, tuple)), 'inplace must be int or tuple'
		assert isinstance(criterion, torch.nn.modules.loss._Loss), 'criterion must be instance of _Loss'
		assert isinstance(transforms, dict) or transforms is None, 'transforms must be instance of dict'
		assert isinstance(optimizer, torch.optim.Optimizer) or optimizer is None, 'parameter must be instance of Optimizer'
		
		self.save_point=save_point
		self.batch_size=batch_size
		self.workers=workers
		self.epochs=epochs
		self.patience=patience
		self.lr=lr
		self.inplace=inplace
		self.criterion = criterion
		self.optimizer = optimizer

		if transforms:
			self.transforms=transforms
		else:
			self.transforms={  #케이스 별 transform 정의
			'train':torchvision.transforms.Compose([torchvision.transforms.Resize(self.inplace), torchvision.transforms.ToTensor()]),
			'valid':torchvision.transforms.Compose([torchvision.transforms.Resize(self.inplace), torchvision.transforms.ToTensor()]),
			'test':torchvision.transforms.Compose([torchvision.transforms.Resize(self.inplace),  torchvision.transforms.ToTensor()])
			}

	def set_transforms(self, transforms:dict):
		assert isinstance(transforms, dict), 'transforms must be a dictionary'
		self.transforms=transforms
	
	def set_optimizer(self, optimizer):
		assert isinstance(optimizer, torch.optim.Optimizer), 'parameter must be instance of Optimizer'
		self.optimizer=optimizer

	def nomalize(self, img:torch.Tensor):
		return img.float()/255.0
	
	def save_log(self, file:Path):
		file = Path(file)
		check_path(file.parent)

		hyper_log=f'''save_point: {self.save_point}
batch_size: {self.batch_size}
epochs: {self.epochs}
patience: {self.patience}
lr: {self.lr}
criterion: {self.criterion}
optimizer: {self.optimizer}
inplace: {self.inplace}
workers: {self.workers}
'''
		
		with open(file,'a',encoding='utf-8') as f:
			f.write(hyper_log)


class ImageDir(data.Dataset):
	def __init__(self, dataset_path, transforms=None):
		self.dataset_path=Path(dataset_path)		
		self.transforms=transforms

		self.img_paths=[]
		self.classes=[]
		self.labels=[]

		label=0
		for subdir in self.dataset_path.iterdir():
			if not subdir.is_dir():
				continue

			for file in subdir.iterdir():
				if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
					continue

				self.img_paths.append(file.name)
				self.labels.append(label)
			self.classes.append(subdir.name)
			label += 1

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		label=self.labels[idx]
		img_file=self.img_paths[idx]

		img_path=self.dataset_path/self.classes[label]/img_file
		img = Image.open(img_path).convert("RGB")

		if self.transforms:
			img=self.transforms(img)
		
		label=torch.tensor(label, dtype=torch.long)

		return img, label


def save_log(save_dir, *metrics):
	with open(save_dir/'log.txt','a') as f:
		f.writelines([line+'\n' for line in metrics])


def get_confusion(outputs:torch.Tensor, labels:torch.Tensor):
	num_classes=outputs.shape[1]

	cm=confusion_matrix(preds=outputs,target=labels,num_classes=num_classes, task='multiclass', threshold=0)
	tp=torch.diag(cm)
	fp=cm.sum(dim=0)-torch.diag(cm)
	fn=cm.sum(dim=1)-torch.diag(cm)
	tn=cm.sum()-fn-fp-tp

	return [value.sum().item() for value in (tp,fp,fn,tn)]


def no_overwrite(path, mode='dir')->Path: #기존 훈련 파일이 덮어써지지 않도록 하는 함수
	path=Path(path)
	match mode:  
		case 'file':  #파일 레벨의 덮어쓰기 방지
			dir_path=path.parent  #경로에서 디렉토리 부분만 가져옴
			file_name,ext=path.stem, path.suffix #파일명과 확장자 분리
			i=1 #파일명에 추가할 숫자
			while path.exists(): #해당 파일이 존재 시 파일명에 숫자를 붙임
				base=f'{file_name}_{i}{ext}'  
				path=dir_path/base
				i += 1
			return path #유니크 경로 반환

		case 'dir': #디렉토리 레벨의 덮어쓰기 방지
			i=1
			path=path/f'{i}' #새로운 디렉토리 경로 생성
			while path.exists(): #만약 해당 디렉토리가 존재하면
				i += 1
				path=path.with_name(f'{i}') #없는 디렉토리가 나올 때 까지 숫자를 증가시키며 적용
			return path
		case _:
			raise FileNotFoundError()


def run_epoch(model:nn.Module, loader:DataLoader, criterion:_WeightedLoss, optimizer:optim.Optimizer, device:torch.device, mode:str, scaler:Optional[GradScaler] = None): #에폭 하나를 실행
	epoch_loss,epoch_acc=.0,.0  #loss, accuracy 초기화

	match mode:
		case 'train':  #훈련 모드시 모델을 훈련 모드로, gradient를 계산
			model.train()
			grad_mode=torch.enable_grad()
			assert scaler, 'train시 scaler는 반드시 존재해야 합니다'
		case 'valid':  #validation모드에선 모델을 추론 모드로, gradient 계산 안함
			model.eval()
			grad_mode=torch.inference_mode()
		case 'test':  #테스트모드에선 inference모드로
			model.eval()
			grad_mode=torch.inference_mode()  #no_grad보다 훨씬 강력한 모드
		case _:  #기타 케이스가 들어오면 에러 발생
			raise ValueError(f'Invalid mode {mode}') 
		
	dataset_size=0
	epoch_tp,epoch_tn,epoch_fp,epoch_fn=0,0,0,0
	for imgs, labels in loader:  #데이터 로더로부터 이미지와 레이블을 읽어오며
		imgs=imgs.to(device); labels=labels.to(device)  #학습 디바이스로 이동
		optimizer.zero_grad()  #옵티마이저의 그래디언트 초기화

		with grad_mode:  #각 모드 하에서 실행
			with autocast(device.type, torch.bfloat16, True, True):
				outputs=model(imgs)  #추론하고
				if type(outputs) != torch.Tensor:
					outputs = outputs.logits
				
				_,preds=torch.max(outputs,1)  #top 1예측값을 가져옴
				loss=criterion(outputs,labels)  #loss 계산
				
				if mode=='train':  #훈련 모드에선 역전파 포함
					scaler.scale(loss).backward()
					scaler.step(optimizer)
					scaler.update()

		batch_size=len(imgs)
		dataset_size += batch_size

		tp,tn,fp,fn=get_confusion(outputs, labels)
		epoch_tp += tp
		epoch_tn += tn
		epoch_fp += fp
		epoch_fn += fn

		epoch_loss += loss.item()*batch_size  #epoch loss에 batch별 loss 가산
		epoch_acc += torch.sum(preds==labels).item()  #accuracy도 동일

	epoch_tp/=dataset_size
	epoch_tn/=dataset_size
	epoch_fp/=dataset_size
	epoch_fn/=dataset_size

	precision=epoch_tp/(epoch_tp+epoch_fp)
	recall=epoch_tp/(epoch_tp+epoch_fn)

	epoch_loss/=dataset_size  #epoch 평균 구하기
	epoch_acc/=dataset_size  #epoch 평균 구하기

	return epoch_loss, epoch_acc, precision, recall #loss와 acc의 평균 반환


def train_valid_run(model:nn.Module, train_loader:DataLoader, valid_loader:DataLoader, hyper_param, save_dir:Union[Path,str]):  #훈련 함수
	save_dir=Path(save_dir); save_dir.mkdir(exist_ok=True,parents=True)

	device:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #사용 가능한 디바이스 확인
	model = model.to(device)  #모델을 디바이스로 이동
	print(device)

	es_count, total_duration=0,0  #early stop 카운트와 총 수행시간을 0으로 초기화
	minimum_loss=float('inf')  #최소 loss를 무한대로 초기화
	best_path=save_dir/'best_weight.pt'  #best weight 경로 설정
	last_path=save_dir/'last_weight.pt'  #last weight 경로 설정

	logs=[]
	train_losses,train_accuracies=[],[]
	valid_losses,valid_accuracies=[],[]
	scaler = GradScaler()
	for epoch in range(1, hyper_param.epochs+1):
		since=time.time()  #에폭 시작 시간
		train_loss, train_accuracy, train_precision, tarin_recall = run_epoch(model, train_loader, hyper_param.criterion, hyper_param.optimizer, device, 'train', scaler)  #훈련 실행
		valid_loss, valid_accuracy, valid_precision, valid_recall = run_epoch(model, valid_loader, hyper_param.criterion, hyper_param.optimizer, device, 'valid', scaler)  #검증 실행

		duration=time.time()-since  #에폭 수행시간 계산
		total_duration += duration  #총 수행시간에 합산

		print(f'epochs: {epoch}/{hyper_param.epochs}, train loss: {train_loss:.4f}, val loss: {valid_loss:.4f}, train accuracy:{train_accuracy:.4f}, val accuracy: {valid_accuracy:.4f}, duration: {duration:.0f}, total duration: {total_duration:.0f}, precision: {valid_precision:.4f}, recall: {valid_recall:.4f}')
		log=f'epochs: {epoch}/{hyper_param.epochs}, train loss: {train_loss}, val loss: {valid_loss}, train accuracy:{train_accuracy}, val accuracy: {valid_accuracy}, train precision: {train_precision}, val precision: {valid_precision}, train recall: {tarin_recall}, val recall: {valid_recall} duration: {duration}, total duration: {total_duration}'
		logs.append(log)

		train_losses.append(train_loss); train_accuracies.append(train_accuracy)
		draw_graph(train_losses,train_accuracies,save_dir/'train_graph.png')

		valid_losses.append(valid_loss); valid_accuracies.append(valid_accuracy)
		draw_graph(valid_losses,valid_accuracies, save_dir/'valid_graph.png')
		
	#early stop
		if minimum_loss<valid_loss:  #검증 로스가 최소치보다 작지 않으면
			es_count += 1  #es count를 증가시킨다
			if hyper_param.patience>0 and es_count>=hyper_param.patience:  #만약 patience가 0보다 크고, es_count가 patience보다 높다면
				torch.save(model.state_dict(),last_path)  #최종 훈련 가중치를 저장하고 학습 종료
				print('early stop')
				break

		else:  #현재 loss가 최소치면
			minimum_loss=valid_loss  #minimum loss를 갱신
			es_count=0  #early stop count를 초기화
			torch.save(model.state_dict(), best_path)  #best weight를 저장
			best_log=log

	#save_point
		if epoch%hyper_param.save_point==0:  #세이브 포인트마다 에폭 저장
			save_point_path=save_dir/f'{epoch}_weight.pt'  #해당 에폭을 파일 명으로 가중치 저장
			torch.save(model.state_dict(), save_point_path) 
			save_log(save_dir,*logs)
			logs=[]
		
	last_log=['\n',f'best: {best_log}',f'last: {log}']
	save_log(save_dir, *logs, *last_log)

	torch.save(model.state_dict(),last_path)
	model.load_state_dict(torch.load(best_path,weights_only=True))


def run_test(model:nn.Module, test_loader:DataLoader, hyper_param, save_dir, device = None):
	if device is None:
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
	model.to(device)
	since=time.time()*1000
	test_loss, test_acc, precision, recall=run_epoch(model, test_loader, hyper_param.criterion, hyper_param.optimizer, device, 'test')
	latency=time.time()*1000-since
	latency_per_img=latency/len(test_loader.dataset)
	print(metrics:=f'Accuracy: {test_acc}, Loss: {test_loss}, Precision: {precision}, Recall: {recall}, Latency: {latency}ms, Latency per img: {latency_per_img}ms')

	if save_dir:
		save_dir=Path(save_dir); save_dir.mkdir(exist_ok=True,parents=True)
		with open(save_dir/'log.txt','a') as f:
			f.write(f'test: {metrics}')


def train_test(model:nn.Module, train_loader:DataLoader, valid_loader:DataLoader, test_loader:DataLoader, hyper_param:TrainConfig, save_dir:Union[Path,str]):  #훈련 및 테스트트 함수
	train_valid_run(model, train_loader, valid_loader, hyper_param, save_dir)
	run_test(model, test_loader, hyper_param, save_dir)


def layer_freeze(model:torch.nn.Module, freeze_until_layer_name = None, freeze_until_layer_num = None):	#until 없으면 전부 freeze
	num = 0
	is_name_match = (lambda name:name.startswith(freeze_until_layer_name)) if freeze_until_layer_name else (lambda _: False)
	for name, param in model.named_parameters():
		num_match = freeze_until_layer_num == num

		if is_name_match(name) or num_match:
			break
		param.requires_grad = False
		num  +=  1


def print_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)


def draw_graph(loss, accuracy, save_path):
	fig = plt.figure()
	ax1 = fig.add_subplot()

	ax1.set_xlabel('Epoch')
	ax1.plot(loss,'b-')
	ax1.set_ylabel('Loss', color='b')

	ax2 = ax1.twinx()
	ax2.plot(accuracy, 'r-', label='Accuracy')
	ax2.set_ylabel('Accuracy', color='r')

	plt.title('Model Graph')
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()