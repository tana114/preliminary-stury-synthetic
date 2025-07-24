"""
MIT License

Copyright (c) 2025 taro nakano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, List, Any, Callable
import codecs
from functools import partial
import json
import csv
import os
import io

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class FileOpenerWrapper(object):
    """
    A wrapper to simplify file opening by allowing partial application of parameters.
    This wrapper enables pre-configuring file opening functions with specific modes and encodings, such as `codecs.open`.
    This allows you to open files by specifying only the filename.

    For example:
        codecs.open(file_name, mode='rb', encoding='utf-8') -> codecs.StreamReaderWriter
        ↓
        open_func = partial(codecs.open, mode='rb', encoding='utf-8')
        ↓
        open_func(file_name) -> codecs.StreamReaderWriter
    """

    def __init__(
            self,
            file_opener: Any = codecs.open,
            **kwargs,
    ):
        """
        Initializes the FileOpenerWrapper with a file opener function and keyword arguments.

        Parameters
        ----------
        file_opener : Any, optional
            The file opening function to wrap, defaults to codecs.open
        **kwargs :
            Keyword arguments to pass to the file opening function.
        """
        self._partial_opener = partial(file_opener, **kwargs)

    def __call__(self, file_name: str) -> Any:
        return self._partial_opener(file_name)


class FileHandler(metaclass=ABCMeta):
    """
    Abstract base class for file handlers.
    Defines the basic interface for reading and writing files.
    """

    def __init__(
            self,
            reader: Optional[FileOpenerWrapper],
            writer: Optional[FileOpenerWrapper],
    ):
        """
        Initializes the FileHandler with optional reader and writer FileOpenerWrapper instances.

        Parameters
        ----------
        reader : Optional[FileOpenerWrapper]
            The FileOpenerWrapper instance for reading files.
        writer : Optional[FileOpenerWrapper]
            The FileOpenerWrapper instance for writing files.
        """
        self._reader = reader
        self._writer = writer

    def read(self, file_name: str):
        if self._reader is None:
            raise AttributeError(
                f"Reader is not defined for {self.__class__.__name__}. Please provide a FileOpenerWrapper.")
        try:
            with self._reader(file_name) as srw:
                return self.read_handling(srw)
        except FileNotFoundError:
            logger.error(f"File '{file_name}' not found.")
            raise
        except OSError as e:
            logger.error(f"OS error occurred trying to read {file_name}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error occurred trying to read {file_name}: {e}")
            raise

    def write(self, data, file_name: str):
        if self._writer is None:
            raise AttributeError(
                f"Writer is not defined for {self.__class__.__name__}. Please provide a FileOpenerWrapper.")
        try:
            with self._writer(file_name) as srw:
                self.write_handling(data, srw)
        except OSError as e:
            logger.error(f"OS error occurred trying to write {file_name}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error occurred trying to write {file_name}: {e}")
            raise

    @abstractmethod
    def read_handling(self, srw: codecs.StreamReader):
        """
        Subclasses must implement this method to handle the actual reading process.
        """
        raise NotImplementedError

    @abstractmethod
    def write_handling(self, data, srw: codecs.StreamWriter):
        """
        Subclasses must implement this method to handle the actual writing process.
        """
        raise NotImplementedError


''' ------ Tools for reading and writing text to files. ------ '''


class JsonlHandler(FileHandler):
    """
    FileHandler subclass for reading and writing JSONL files (JSON Lines format).
    Reads each line as a JSON object and returns a list of dictionaries.
    Writes a list of dictionaries to the file, with each dictionary as a JSON object on a new line.
    """

    def __init__(self):
        reader = FileOpenerWrapper(codecs.open, mode='r', encoding='utf-8')
        writer = FileOpenerWrapper(codecs.open, mode='w', encoding='utf-8')
        super().__init__(reader, writer)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> List[Dict]:
        # jsonl_data = [json.loads(l) for l in srw.readlines()]
        # return jsonl_data
        jsonl_data = (json.loads(l) for l in srw)
        return list(jsonl_data)

    def write_handling(self, data: List[Dict], srw: codecs.StreamReaderWriter):
        # data_cl = [json.dumps(d, ensure_ascii=False) + "\n" for d in data]
        # srw.writelines(data_cl)
        for d in data:
            json_str = json.dumps(d, ensure_ascii=False) + "\n"
            srw.write(json_str)


class JsonHandler(FileHandler):
    """
    FileHandler subclass for reading and writing JSON files.
    Reads the entire file as a JSON object and returns a dictionary.
    Writes a dictionary to the file as a JSON object.
    """

    def __init__(self):
        reader = FileOpenerWrapper(codecs.open, mode='r', encoding='utf-8')
        writer = FileOpenerWrapper(codecs.open, mode='w', encoding='utf-8')
        super().__init__(reader, writer)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> Dict:
        try:
            return json.load(srw)
        except json.decoder.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            # return {}
            raise

    def write_handling(self, data: Dict, srw: codecs.StreamReaderWriter):
        json.dump(data, srw, ensure_ascii=False, indent=2)


class CsvHandler(FileHandler):
    """
    FileHandler subclass for reading and writing CSV files.
    Supports reading CSV files into a list of rows, where each row is a list of optional values,
    and writing lists of rows to CSV files with specified encoding.
    """

    def __init__(self):
        reader = FileOpenerWrapper(codecs.open, mode='r', encoding='utf-8')
        writer = FileOpenerWrapper(codecs.open, mode='w', encoding='utf-8')
        super().__init__(reader, writer)

    def read_handling(self, srw: codecs.StreamReaderWriter) -> List[List[Optional[Any]]]:
        reader = csv.reader(srw)
        line_list = []
        for row in reader:
            line_list.append(row)
        return line_list

    def write_handling(self, data: List[List[Optional[Any]]], srw: codecs.StreamReaderWriter):
        writer = csv.writer(srw)
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    """
    python -m util.file_tools
    """

    from logging import DEBUG, INFO, basicConfig

    basicConfig(level=INFO)

    ''' csv read and write '''
    csv_h = CsvHandler()
    csv_file = "./data/sample_write.csv"

    data_list = [
        ["hoge", 3],
        ["fuga", 4],
    ]

    csv_h.write(data_list, csv_file)
    csv_data = csv_h.read(csv_file)

    print(csv_data)

    """ test for jsonl """
    jl_h = JsonlHandler()
    jl_file = "./data/sample_write.jsonl"

    dict_list = [
        {
            "id": "task_10",
            "instruction": "朝食",
        },
        {
            "id": "task_11",
            "instruction": "ビル",
        },
        {
            "id": "task_12",
            "instruction": "人物",
        }
    ]

    jl_h.write(dict_list, file_name=jl_file)

    jl_data = jl_h.read(jl_file)
    print(jl_data)

    """ test for json """
    j_h = JsonHandler()
    j_file = "./data/sample_write.json"

    dict_data = {"data": dict_list}

    j_h.write(dict_data, file_name=j_file)

    j_data = j_h.read(j_file)
    print(j_data)

    # """ test for TextOpenerHandler """
    # text_opener = TextOpenerHandler()
    # t_data = text_opener.read("./data/test.txt")
    #
    # """ test for MarkItDownHandler """
    # mid_opener = MarkItDownHandler()
    #
    # # mid_data = mid_opener.read("./data/ppt_basic.pptx")
    # # mid_data = mid_opener.read("./data/word_basic.docx")
    # mid_data = mid_opener.read("./data/pdf_basic.pdf")
    # print(mid_data)
