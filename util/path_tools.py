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

from typing import Optional, List, Union, Generator
from typing import cast
from pathlib import Path
from abc import ABCMeta, abstractmethod
import itertools

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class PathHandler(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, **kwargs) -> object:
        """
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def dir_path_obj(dir_path: str) -> Path:
        """
        Validate and return a Path object for a directory.
        Raises an error if the path is invalid, does not exist, or is not a directory.
        """
        if not dir_path:
            raise ValueError("Directory path is empty. Please specify a valid path.")
        p_dir = Path(dir_path).resolve()
        if not p_dir.exists():
            raise FileNotFoundError(f"The specified directory '{str(p_dir)}' does not exist.")
        if not p_dir.is_dir():
            raise NotADirectoryError(f"The path '{str(p_dir)}' is not a directory.")
        return p_dir

    @staticmethod
    def file_path_obj(file_path: str) -> Path:
        """
        Validate and return a Path object for a file.
        Raises an error if the path is invalid, does not exist, or is not a file
        """
        if not file_path:
            raise ValueError("File path is empty. Please specify a valid path.")
        p_file = Path(file_path).resolve()
        if not p_file.exists():
            raise FileNotFoundError(f"The specified file '{str(p_file)}' does not exist.")
        if not p_file.is_file():
            raise IsADirectoryError(f"The path '{str(p_file)}' is not a file.")
        return p_file


class SuffixFilteredPathBuilder(PathHandler):
    def __init__(self, file_suffix: Union[str, List[str]], recursive: bool = False):
        """
        Get a list of files with specified suffix(es) in a directory.
        Initialize with suffix(es) to search for and recursion option.

        Note: The search will match files ending with the specified suffix(es).

        Parameters
        ----------
        file_suffix : str or list of str
            The suffix or list of suffixes to match files (e.g., '.py', '.txt').
        recursive : bool, optional
            Whether to search recursively or not. Default is False.
        """
        self._recursive = recursive
        self._file_suffix_list = []

        if isinstance(file_suffix, str):
            self._file_suffix_list.append(file_suffix)
        elif isinstance(file_suffix, list):
            self._file_suffix_list = file_suffix
        else:
            raise TypeError("file_suffix must be a str or list of str")

    def __call__(self, src_dir: Optional[str] = None) -> Generator[str, None, None]:
        """
        Search for files with the specified suffix(es) in the given directory.

        Parameters
        ----------
        src_dir : str, optional
            Directory path to search. Defaults to current working directory if None.

        Returns
        -------
        Generator[str, None, None]
            A generator yielding absolute file paths matching the suffix(es).
        """
        files_gen = itertools.chain()  # empty generator
        p_src_dir = Path.cwd() if src_dir is None else self.dir_path_obj(src_dir)

        for suffix in self._file_suffix_list:
            wild_card = "*" + suffix
            if self._recursive:
                p_file_iter = p_src_dir.rglob(wild_card)
            else:
                p_file_iter = p_src_dir.glob(wild_card)
            p_file_str_iter = (str(p.resolve()) for p in p_file_iter)
            files_gen = itertools.chain(files_gen, p_file_str_iter)
        # Explicitly cast the type hint. chain -> Generator
        return cast(Generator[str, None, None], files_gen)


class OutputPathCreator(PathHandler):
    def __init__(
            self,
            out_suffix: str,
            output_dir: Optional[str] = None,
            add_stem: Optional[str] = None,
            avoid_dup: bool = False,
            prefix_to_avoid: str = '-dup',
    ):
        """
        Generate a new output file path based on input file path, output directory, and suffix.

        Parameters
        ----------
        out_suffix : str
            The suffix (file extension) for the generated output file path. Example: '.txt'. Must start with a dot.
        output_dir : str, optional
            The directory where the output file will be saved. Must be specified.
        add_stem : str, optional
            String to append to the filename stem to differentiate output files.
        avoid_dup : bool, optional
            If True, appends a prefix to the filename to avoid duplication.
        prefix_to_avoid: str
            Prefix used to avoid duplicate filenames.
        """
        if not out_suffix:
            raise ValueError("Specify the suffix of the output file!")

        self._output_suffix = out_suffix
        self._add_stem = add_stem
        self._p_out_dir = Path.cwd() if output_dir is None else Path(output_dir)
        self._avoid_dup = avoid_dup
        self._prefix_to_avoid = prefix_to_avoid

    def __call__(self, input_file_path: str, add_stem: Optional[str] = None) -> str:
        """
        Generate the output file path based on input file path and optional additional stem.

        Parameters
        ----------
        input_file_path : str
            Path to the input file. Must be an existing file.
        add_stem : str, optional
            Additional string to append to the filename stem.

        Returns
        -------
        str
            The generated output file path.
        """

        p_file = self.file_path_obj(input_file_path)

        self._p_out_dir.mkdir(parents=True, exist_ok=True)

        p_tmp_file = p_file.with_suffix(self._output_suffix)

        combined_add_stem = (self._add_stem or "") + (add_stem or "")  # Combine instance and call add_stem
        if combined_add_stem:
            out_file_name = p_tmp_file.stem + combined_add_stem + p_tmp_file.suffix
        else:
            out_file_name = p_tmp_file.name

        if self._avoid_dup:
            out_file_name = self._generate_unique_name(str(self._p_out_dir), out_file_name)

        p_out_file = self._p_out_dir.joinpath(out_file_name)

        return str(p_out_file.resolve())

    def _generate_unique_name(self, out_dir: str, base_name: str, count: int = 0) -> str:
        """
        Generate a unique filename within the specified output directory to avoid overwriting existing files.

        Parameters
        ----------
        out_dir : str
            The target directory for output destination.
        base_name : str
            The base filename, including extension.
        count : int
            A counter to append to the filename to ensure uniqueness, used in recursive calls.

        Returns
        -------
        str
            A filename that does not exist in the output directory.
        """
        p_out_dir = Path(out_dir)
        if not p_out_dir.is_dir():
            raise NotADirectoryError(f"The path '{str(p_out_dir)}' is not a directory.")

        base_path = Path(base_name)
        if count == 0:
            candidate_path = p_out_dir / base_path.name
        else:
            # The filename is prefixed with a counter to avoid duplication.
            candidate_name = f"{base_path.stem}{self._prefix_to_avoid}{count}{base_path.suffix}"
            candidate_path = p_out_dir / candidate_name
        if candidate_path.exists():
            return self._generate_unique_name(str(p_out_dir), base_name, count + 1)
        else:
            return str(candidate_path.name)


if __name__ == "__main__":
    '''
    Simple test for PathHandler class

    python -m util.path_tools
    '''
    from logging import DEBUG, INFO, basicConfig

    basicConfig(level=INFO)

    # Test GetFileListBySuffix
    fp_picker = SuffixFilteredPathBuilder(['.py', ], recursive=True)
    filtered_files_gen: Generator[str, None, None] = fp_picker()  # search on current directory
    first_file = next(filtered_files_gen)
    print(first_file)
    print(list(filtered_files_gen))

    # Test OutputFilePathGenerator
    ofp_getter = OutputPathCreator(out_suffix='.txt')
    # out_dir = 'sample_dir'
    # ofp_getter = OutputPathCreator(out_suffix='.txt', output_dir=out_dir, avoid_dup=True)
    output_file_path: str = ofp_getter(first_file)
    print(output_file_path)

