import PIL.Image as Image
import PIL.ImageFile as ImageFile
from pathlib import Path
import threading, os, piexif, shutil, json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections.abc import Sequence, Iterable
from typing import Any
from functools import wraps

ImageFile.LOAD_TRUNCATED_IMAGES=True


class ParallelExecutor:
    def __init__(self, aggregator = None) -> None:
        self.aggregator = aggregator
    
    def __call__(self, func) -> Any:
            @wraps(func)
            def wrapper(iterable:Iterable, *args, n_workers:int = 0, use_processing:bool = False, **kwargs) -> Any:
                if not isinstance(iterable, Iterable):
                    raise TypeError('입력은 Iterable이여야 합니다.')
                sequence = iterable if isinstance(iterable, Sequence) else list(iterable)

                requested_workers = n_workers if n_workers > 0 else os.cpu_count() or 1
                actual_workers = max(1 , min(requested_workers, len(sequence)))

                chunks = equal_split(sequence, actual_workers)
                executor_class = ProcessPoolExecutor if use_processing else ThreadPoolExecutor
                
                results = [[None] for _ in range(actual_workers)]
                with executor_class(max_workers = actual_workers) as executor:
                    futures = {executor.submit(func, chunk, *args, **kwargs) : i for i, chunk in enumerate(chunks)}
                    try:
                        for future in as_completed(futures):
                            idx = futures[future]
                            results[idx] = future.result()
                    
                    except Exception as e:
                        for k in futures.keys():
                            k.cancel()
                        raise e

                    else:
                        if self.aggregator:
                            aggregated_result = self.aggregator(results)
                        else:
                            try:
                                aggregated_result = self.default_aggregator(results)
                            except Exception as e:
                                raise e
                        return aggregated_result
            return wrapper


    def shallow_flatten(self, iterable:list):
        shallow_flattened = []
        for item in iterable:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub_item in item:
                    shallow_flattened.append(sub_item)
            else:
                shallow_flattened.append(item)
        return shallow_flattened
    

    def default_aggregator(self, results:list[Any]) -> Any:
        if not len(results):
            return results
        
        if isinstance(results[0], Sequence):
            aggregated_result = self.shallow_flatten(results)
        elif isinstance(results[0], dict):
            aggregated_result = {}
            for e in results:
                aggregated_result |= e
        elif isinstance(results[0], set):
            aggregated_result = set()
            for e in results:
                aggregated_result |= e
        else:
            raise TypeError('default_aggtegator에서는 합칠 수 없는 자료형입니다')

        return aggregated_result


def equal_split(items:Sequence, n_chunk:int) -> Sequence:
    sequence_len = len(items)
    
    if not sequence_len:
        return [items]

    n_chunk = n_chunk if (n_chunk != 0) and (n_chunk <= sequence_len) else sequence_len
    chunk_size, residual = divmod(sequence_len, n_chunk)
    
    chunks = []
    start = 0
    for _ in range(n_chunk):
        end = start + chunk_size

        if residual > 0:
            end += 1
            residual -= 1

        chunk = items[start:end]
        chunks.append(chunk)
        start = end

    return chunks


def shallow_flatten(iterable:Iterable):
    shallow_flattened = []
    for item in iterable:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for sub_item in item:
                shallow_flattened.append(sub_item)
        else:
            shallow_flattened.append(item)
    return shallow_flattened


def is_rgb(img_path:Path):
    with Image.open(img_path) as img:
        return img.mode == 'RGB'


def separate_non_rgb(directory_list:list, root:Path):
    separation_dir = root/'non_rgb'
    for directory in directory_list:
        directory = Path(directory)
        for file in directory.iterdir():
            if is_rgb(file):
                continue
            
            separation_dir.mkdir(exist_ok=True)
            name = file.stem
            extension = file.suffix
            file_path = separation_dir/directory/file.name

            i=1
            while file_path.exists():
                file_path = separation_dir/directory/f'{name}({i}){extension}'
                i+=1
            file.rename(file_path)


def get_files(p:Path):
    if p.is_file():
        return [p]
    
    files = []
    for child in p.iterdir():
        if child.is_dir():
            files.extend(get_files(child))
        elif child.is_file():
            files.append(child)
    
    return files


def dataset_split(input_dir:Path|str, val_rate:float, test_rate:float):
    input_dir = Path(input_dir)
    assert val_rate + test_rate <= 1, '합계 비율이 1 이하여야 합니다'
    assert input_dir.is_dir(), '입력및 출력 디렉토리는 반드시 디렉토리여야합니다'

    for sub_dir in input_dir.iterdir():
        files = list(sub_dir.iterdir())
        n_file = len(files)
        n_val = int(n_file * val_rate)
        n_test = int(n_file * test_rate)
        
        val_files = files[:n_val]
        test_files = files[n_val:n_val+n_test]
        train_files = files[n_val+n_test:]

        for file_subset, group in ((val_files, 'valid'), (test_files, 'test'), (train_files, 'train')):
            current_dir = input_dir/group/sub_dir.name  
            current_dir.mkdir(exist_ok=True, parents=True)
            for file in file_subset:
                shutil.move(file, current_dir/file.name)
        
        if sub_dir.name not in ('train', 'test', 'valid'):
            os.rmdir(sub_dir)


def chk_corrupt(root:Path, dirlist):
    separtion_dir = root/'corrupt_img'

    for sub_dir in dirlist:
        sub_dir = Path(sub_dir)
        for file in sub_dir.iterdir():
            try:
                with Image.open(file) as img:
                    img.verify()
                    exif_data = img.info.get('exif')
                    if exif_data:
                        piexif.load(exif_data)
            except:
                sepration_sub_dir = separtion_dir/sub_dir.name
                sepration_sub_dir.mkdir(exist_ok=True, parents=True)
                new_path = sepration_sub_dir/file.name
                
                file.rename(new_path)


def prune_excess_files(root):
    file_lens = []
    for subdir in root.iterdir():
        file_lens.append(len(list(subdir.iterdir())))

    minimum_files = min(file_lens)

    for subdir in root.iterdir():
        files = list(subdir.iterdir())
        outlier = files[minimum_files:]

        for f in outlier:
            f.unlink()


def organize_files_by_json(root):
    root = Path(root)
    with open('files.json',  encoding='utf-8') as f:
        file_structure = json.load(f)

    for subset, calss_structure in file_structure.items():
        subset_dir = root / subset
        subset_dir.mkdir(exist_ok = True, parents = True)
        for class_name, files in calss_structure.items():
            class_dir = subset_dir / class_name
            class_dir.mkdir(exist_ok = 1, parents = 1)
            for file in files:
                shutil.move(root/class_name/file, class_dir/file)


if __name__ == '__main__':
    pass