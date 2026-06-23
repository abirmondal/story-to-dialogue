"""
load_data_soda.py

This module contains functions to load and preprocess the SODA dataset for story-to-dialogue conversion tasks.
The dataset is created by Kim et al. (2023) and is available at https://huggingface.co/datasets/allenai/soda.
"""

import json
import random
from transformers import AutoTokenizer
from datasets import load_dataset, load_dataset_builder, DatasetDict, Dataset
from config.dir import SODA_HF_REPO, SODA_BENGALI_HF_REPO
from config.dialogue_special_tokens import DIALOGUE_END_TOKEN, DEFAULT_SEPARATOR_TOKEN
from config.llama_config import get_chat_template

class SODADataLoader:
    """
    Class to load and preprocess the SODA dataset.
    """
    def __init__(
            self,
            use_bengali_soda: bool = False,
            data_types: list[str] = ['train', 'test', 'validation'],
            use_features: list[str] = ['narrative', 'dialogue', 'speakers'],
            percent_of_all_splits: float | None = None,
            samples_per_split: int | None = None,
            join_narrative_and_speakers: bool = False,
            join_with: str | None = None,
            unroll_dialogue_and_speakers: bool = False,
            use_eos_as_eod: bool = False,
            separator_token: str = DEFAULT_SEPARATOR_TOKEN,
            join_dialogue_and_speakers: bool = False,
            add_characters_in_narrative: bool = False,
            add_turns_count_in_narrative: bool = False,
            min_story_length: int | None = None,
            max_story_length: int | None = None,
            min_dialogue_length: int | None = None,
            max_dialogue_length: int | None = None,
            show_dataset_info_after_load: bool = True,
            keep_speakers_col: bool = False
        ) -> None:
        """
        Initializes the SODADataLoader with specified parameters.

        Args:
            use_bengali_soda (bool): If `True`, loads the SODA Bengali dataset. Default is `False`.
            data_types (list): List of dataset splits to load. Options are `train`, `test`, `validation`.
            use_features (list): List of features to retain from the dataset. For all features, use `['all']`.
            percent_of_all_splits (float | None): Percentage of each split to load (between 0 and 100). Default is `None`, which loads the full splits.
            samples_per_split (int | None): Number of samples to load per split. If specified, overrides `percent_of_all_splits`. Default is `None`.
            join_narrative_and_speakers (bool): If `True`, joins the `narrative` and `speakers` features into a single feature.
            join_with (str | None): String to use for joining `narrative` and `speakers` if `join_narrative_and_speakers` is `True`.
            unroll_dialogue_and_speakers (bool): If `True`, creates separate examples for each dialogue turn with corresponding speaker in the narrative. Default is `False`. If `True`, `join_dialogue_and_speakers` must be `False`.
            use_eos_as_eod (bool): If `True`, uses the end-of-sequence token as the dialogue end token. Default is `False`. It can only be `True` when `unroll_dialogue_and_speakers` is `True`.
            separator_token (str | None): The separator token to use to join features. Default is `DEFAULT_SEPARATOR_TOKEN` set in `config/dialogue_special_tokens.py`.
            join_dialogue_and_speakers (bool): If `True`, joins the `dialogue` and `speakers` features into a single feature.
            add_characters_in_narrative (bool): If `True`, adds character information to the `narrative` feature.
            add_turns_count_in_narrative (bool): If `True`, adds turn count to the `narrative` feature.
            min_story_length (int | None): Minimum number of words in the `narrative` feature to retain an example. If `None`, no minimum is applied.
            max_story_length (int | None): Maximum number of words in the `narrative` feature to retain an example. If `None`, no maximum is applied.
            min_dialogue_length (int | None): Minimum number of words in the `dialogue` feature to retain an example. If `None`, no minimum is applied.
            max_dialogue_length (int | None): Maximum number of words in the `dialogue` feature to retain an example. If `None`, no maximum is applied.
            show_dataset_info_after_load (bool): If `True`, displays dataset information including feature details after loading. Default is `True`.
            keep_speakers_col (bool): If `True`, retains the `speakers` column in the dataset even after joining with `narrative` or `dialogue`. Default is `False`.
        """
        if data_types is None or len(data_types) == 0:
            raise ValueError("data_types must be a non-empty list containing any of 'train', 'test', 'validation'.")
        # Check if data_types value is valid
        valid_splits = {'train', 'test', 'validation'}
        for split in data_types:
            if split not in valid_splits:
                raise ValueError(f"Invalid data_type '{split}'. Valid options are 'train', 'test', 'validation'.")
        if use_features is None or len(use_features) == 0:
            raise ValueError("use_features must be a non-empty list of feature names or ['all'].")
        if 'all' in use_features and len(use_features) > 1:
            raise ValueError("If 'all' is specified in use_features, it must be the only entry.")
        if percent_of_all_splits is not None:
            if percent_of_all_splits <= 0 or percent_of_all_splits > 100:
                raise ValueError("percent_of_all_splits must be between 1 and 100.")
        if unroll_dialogue_and_speakers and join_dialogue_and_speakers:
            raise ValueError("Only one of unroll_dialogue_and_speakers or join_dialogue_and_speakers can be True.")
        if unroll_dialogue_and_speakers and ('narrative' not in use_features or 'dialogue' not in use_features or 'speakers' not in use_features or 'all' in use_features):
            raise ValueError(
                "To unroll dialogue and speakers, all of 'narrative', 'dialogue', and 'speakers' must be in use_features or use_features must be ['all'].")
        if use_eos_as_eod and unroll_dialogue_and_speakers == False:
            raise ValueError("use_eos_as_eod can only be True when unroll_dialogue_and_speakers is True.")
        if join_narrative_and_speakers and ('speakers' not in use_features or 'narrative' not in use_features or 'all' in use_features):
            raise ValueError(
                "To join narrative and speakers, both 'narrative' and 'speakers' must be in use_features or use_features must be ['all'].")
        if join_narrative_and_speakers and (join_with is None):
            raise ValueError("join_with must be a non-empty string when join_narrative_and_speakers is True.")
        if join_dialogue_and_speakers and ('speakers' not in use_features or 'dialogue' not in use_features or 'all' in use_features):
            raise ValueError(
                "To join dialogue and speakers, both 'dialogue' and 'speakers' must be in use_features or use_features must be ['all'].")
        if join_dialogue_and_speakers and join_narrative_and_speakers:
            raise ValueError("Only one of join_narrative_and_speakers or join_dialogue_and_speakers can be True.")
             
        # validate story length bounds
        if min_story_length is not None and min_story_length <= 0:
            raise ValueError("min_story_length must be a positive integer or None")
        if max_story_length is not None and max_story_length <= 0:
            raise ValueError("max_story_length must be a positive integer or None")
        if min_story_length is not None and max_story_length is not None and min_story_length > max_story_length:
            raise ValueError("min_story_length cannot be greater than max_story_length")
        
        # validate dialogue length bounds
        if min_dialogue_length is not None and min_dialogue_length <= 0:
            raise ValueError("min_dialogue_length must be a positive integer or None")
        if max_dialogue_length is not None and max_dialogue_length <= 0:
            raise ValueError("max_dialogue_length must be a positive integer or None")
        if min_dialogue_length is not None and max_dialogue_length is not None and min_dialogue_length > max_dialogue_length:
            raise ValueError("min_dialogue_length cannot be greater than max_dialogue_length")

        # store small config params centrally (avoid setting as many self.* attributes)
        self.dataset_info: dict = {
            'params': {
                'use_bengali_soda': use_bengali_soda,
                'data_types': data_types,
                'use_features': use_features,
                'percent_of_all_splits': percent_of_all_splits,
                'samples_per_split': samples_per_split,
                'unroll_dialogue_and_speakers': unroll_dialogue_and_speakers,
                'use_eos_as_eod': use_eos_as_eod,
                'separator_token': separator_token,
                'join_narrative_and_speakers': join_narrative_and_speakers,
                'join_dialogue_and_speakers': join_dialogue_and_speakers,
                'add_characters_in_narrative': add_characters_in_narrative,
                'add_turns_count_in_narrative': add_turns_count_in_narrative,
                'min_story_length': min_story_length,
                'max_story_length': max_story_length,
                'min_dialogue_length': min_dialogue_length,
                'max_dialogue_length': max_dialogue_length,
                'join_with': join_with
            },
            'splits': {}
        }

        # load, filter and preprocess using local params and dataset_info
        dataset = self.__load_data(splits=data_types, features=use_features, percent_of_all_splits=percent_of_all_splits, samples_per_split=samples_per_split, use_bengali_soda=use_bengali_soda)
        dataset = self.__filter_by_story_length(dataset, min_story_length=min_story_length, max_story_length=max_story_length)
        dataset = self.__filter_by_dialogue_length(dataset, min_dialogue_length=min_dialogue_length, max_dialogue_length=max_dialogue_length)
        self.dataset = self.__preprocess_data(
            dataset=dataset,
            unroll_dialogue_and_speakers=unroll_dialogue_and_speakers,
            use_eos_as_eod=use_eos_as_eod,
            separator_token=separator_token,
            join_narrative_and_speakers=join_narrative_and_speakers,
            join_with=join_with,
            join_dialogue_and_speakers=join_dialogue_and_speakers,
            add_characters_in_narrative=add_characters_in_narrative,
            add_turns_count_in_narrative=add_turns_count_in_narrative,
            keep_speakers_col=keep_speakers_col
        )

        # populate minimal metadata (counts, columns). heavy stats are computed lazily on demand
        self._populate_dataset_info()
        if show_dataset_info_after_load:
            self.show_dataset_info(show_features=True)

    def __load_data(
            self,
            splits: list[str],
            features: list[str],
            percent_of_all_splits: float | None = None,
            samples_per_split: int | None = None,
            use_bengali_soda: bool = False
        ) -> DatasetDict:
        """
        Loads the SODA dataset from the Hugging Face repository.
        """
        dataset = {}
        repo = SODA_BENGALI_HF_REPO if use_bengali_soda else SODA_HF_REPO
        rename_map = {
            'translated_narrative': 'narrative',
            'translated_dialogue': 'dialogue',
            'translated_speakers': 'speakers'
        }

        for split in splits:
            if samples_per_split is not None:
                split_str = f"[:{samples_per_split}]"
            elif percent_of_all_splits is not None:
                ds_builder = load_dataset_builder(repo)
                total_num_samples = ds_builder.info.splits[split].num_examples
                num_samples_to_load = int((percent_of_all_splits / 100) * total_num_samples)
                split_str = f"[:{num_samples_to_load}]"
            else:
                split_str = ""
            dataset[split] = load_dataset(repo, split=f"{split}{split_str}")
        dataset = DatasetDict(dataset)
        ds_keys = list(dataset.keys())

        for split in ds_keys:
            if use_bengali_soda:
                for old_name, new_name in rename_map.items():
                    if old_name in dataset[split].column_names:
                        dataset[split] = dataset[split].rename_column(old_name, new_name)

        for split in ds_keys:
            if 'all' not in features:
                dataset[split] = dataset[split].remove_columns([col for col in dataset[split].column_names if col not in features])

        return dataset

    def __preprocess_data(
            self,
            dataset: DatasetDict,
            unroll_dialogue_and_speakers: bool = False,
            use_eos_as_eod: bool = False,
            separator_token: str = DEFAULT_SEPARATOR_TOKEN,
            join_narrative_and_speakers: bool = False,
            join_with: str | None = None,
            join_dialogue_and_speakers: bool = False,
            add_characters_in_narrative: bool = False,
            add_turns_count_in_narrative: bool = False,
            keep_speakers_col: bool = False
        ) -> DatasetDict:
        """
        Preprocesses the SODA dataset by selecting specified splits and features, and optionally joining features.

        Returns:
            DatasetDict: A dictionary containing the specified splits of the dataset.
        """
        processed_splits = {}

        for split in self.dataset_info['params']['data_types']:
            if split in dataset:
                split_data = dataset[split]

                if unroll_dialogue_and_speakers:
                    def unroll_dialogue_speakers(example):
                        """
                        Transforms a single dataset example with full lists into multiple
                        examples, one for each dialogue turn, with history.
                        """
                        new_narratives = []
                        new_dialogues = []
                        new_speakers = []

                        if use_eos_as_eod:
                            # Append an extra turn with the last speaker and empty dialogue if using EOS as EOD
                            example['speakers'][0].append(example['speakers'][0][-2])
                            example['dialogue'][0].append("")

                        for narrative, speakers, utterances in zip(example['narrative'], example['speakers'], example['dialogue']):
                            # Initialize the context with the base narrative
                            context = narrative + separator_token

                            # Iterate through each speaker-dialogue pair
                            for speaker, utterance in zip(speakers, utterances):
                                # 1. Create the new example for this turn
                                # The narrative is the context *before* this turn, plus the new speaker
                                new_narr = context + f"{speaker}:"

                                # The dialogue is the current utterance
                                if speaker == speakers[-1] and utterance == utterances[-1]:
                                    if use_eos_as_eod:
                                        new_diag = utterance
                                    else:
                                        new_diag = utterance + DIALOGUE_END_TOKEN
                                else:
                                    new_diag = utterance + separator_token

                                new_narratives.append(new_narr)
                                new_dialogues.append(new_diag)
                                new_speakers.append(list(set(speakers)))

                                # 2. Update the context for the *next* turn
                                # The context now includes what was just said
                                context += f"{speaker}: {utterance}{separator_token}"

                        return {"narrative": new_narratives, "dialogue": new_dialogues, "speakers": new_speakers}
                    split_data = split_data.map(unroll_dialogue_speakers, desc=f"Unrolling dialogue and speakers for {split} split", batched=True)
                    if not keep_speakers_col:
                        split_data = split_data.remove_columns(['speakers']) 
                
                if join_narrative_and_speakers:
                    def join_narrative_speakers(example):
                        example['narrative'] = f"{example['narrative']}{join_with}{example['speakers']}"
                        example['speakers'] = list(set(example['speakers']))
                        return example
                    split_data = split_data.map(join_narrative_speakers, desc=f"Joining narrative and speakers for {split} split")
                    if not keep_speakers_col:
                        split_data = split_data.remove_columns(['speakers'])
                
                if add_characters_in_narrative:
                    def add_characters(example):
                        characters = set(example['speakers'])
                        characters_str = "Characters: " + ", ".join(characters) + ". "
                        example['narrative'] = example['narrative'] + "\n" + characters_str
                        return example
                    split_data = split_data.map(add_characters, desc=f"Adding characters to narrative for {split} split")

                if add_turns_count_in_narrative:
                    def add_turns_count(example):
                        num_turns = len(example['dialogue'])
                        turns_str = f"Dialogue turns: {num_turns}. "
                        example['narrative'] = example['narrative'] + "\n" + turns_str
                        return example
                    split_data = split_data.map(add_turns_count, desc=f"Adding turn count to narrative for {split} split")

                if join_dialogue_and_speakers:
                    def join_dialogue_speakers(example):
                        # create a single string where each utterance is prefixed by its speaker
                        joined_lines = []
                        for utterance, speaker in zip(example['dialogue'], example['speakers']):
                            joined_lines.append(f"{speaker}: {utterance}")
                        # convert to a single string (separated by newlines) so it can be passed to models
                        example['dialogue'] = "\n".join(joined_lines)
                        example['speakers'] = list(set(example['speakers']))
                        return example
                    split_data = split_data.map(join_dialogue_speakers, desc=f"Joining dialogue and speakers for {split} split")
                    if not keep_speakers_col:
                        split_data = split_data.remove_columns(['speakers'])

                processed_splits[split] = split_data

        return DatasetDict(processed_splits)

    def __get_num_words_in_story_batch(self, batch: list) -> list:
        """
        Computes the number of words in stories in a given batch.

        Args:
            batch (list): A batch of stories.
        
        Returns:
            list: A list of word counts for each story in the batch.
        """
        if not batch:
            return {"story_word_count": []}

        return {"story_word_count": [len(entry.split()) for entry in batch['narrative']]}

    def __get_num_words_in_dialogue_batch(self, batch: list) -> list:
        """
        Computes the number of words in dialogues in a given batch.

        Args:
            batch (list): A batch of dialogues.

        Returns:
            list: A list of word counts for each dialogue in the batch.
        """
        if not batch:
            return {"dialogue_word_count": []}

        counts = []
        for entry in batch['dialogue']:
            # if dialogues were joined with speakers they will be a single string
            if isinstance(entry, str):
                counts.append(len(entry.split()))
            else:
                # otherwise expect a list of utterances
                counts.append(len(" ".join(entry).split()))

        return {"dialogue_word_count": counts}

    def __filter_by_story_length(self, dataset: DatasetDict, min_story_length: int | None = None, max_story_length: int | None = None) -> DatasetDict:
        """
        Filters examples in the dataset splits based on the number of words in the `narrative` feature.

        Args:
            dataset (DatasetDict): The dataset to filter.
            min_story_length (int | None): Minimum number of words (inclusive). If None, no minimum applied.
            max_story_length (int | None): Maximum number of words (inclusive). If None, no maximum applied.

        Returns:
            DatasetDict: A new DatasetDict containing only examples within the specified bounds.
        """
        # If neither bound is provided, return dataset unchanged
        if min_story_length is None and max_story_length is None:
            return dataset

        processed = {}

        def _within_bounds(example):
            # If narrative isn't present, exclude the example
            if 'narrative' not in example or example['narrative'] is None:
                return False
            count = len(example['narrative'].split())
            if min_story_length is not None and count < min_story_length:
                return False
            if max_story_length is not None and count > max_story_length:
                return False
            return True

        for split, data in dataset.items():
            # Only attempt to filter splits that have 'narrative'
            if 'narrative' in data.column_names:
                filtered = data.filter(lambda example: _within_bounds(example), desc=f"Filtering {split} split by story length")
                processed[split] = filtered
            else:
                # keep as-is when no narrative field to evaluate
                processed[split] = data

        return DatasetDict(processed)
    
    def __filter_by_dialogue_length(self, dataset: DatasetDict, min_dialogue_length: int | None = None, max_dialogue_length: int | None = None) -> DatasetDict:
        """
        Filters examples in the dataset splits based on the number of words in the `dialogue` feature.
        The filter is on each individual dialogue and not the total words across all dialogues in a sample.

        Args:
            dataset (DatasetDict): The dataset to filter.
            min_dialogue_length (int | None): Minimum number of words (inclusive). If None, no minimum applied.
            max_dialogue_length (int | None): Maximum number of words (inclusive). If None, no maximum applied.

        Returns:
            DatasetDict: A new DatasetDict containing only examples within the specified bounds.
        """
        # If neither bound is provided, return dataset unchanged
        if min_dialogue_length is None and max_dialogue_length is None:
            return dataset

        processed = {}

        def _within_bounds(example):
            # If dialogue isn't present, exclude the example
            if 'dialogue' not in example or example['dialogue'] is None:
                return False
            dialogue_lengths = list(map(len, example['dialogue']))
            if min_dialogue_length is not None and min(dialogue_lengths) < min_dialogue_length:
                return False
            if max_dialogue_length is not None and max(dialogue_lengths) > max_dialogue_length:
                return False
            return True

        for split, data in dataset.items():
            # Only attempt to filter splits that have 'dialogue'
            if 'dialogue' in data.column_names:
                filtered = data.filter(lambda example: _within_bounds(example), desc=f"Filtering {split} split by dialogue length")
                processed[split] = filtered
            else:
                # keep as-is when no dialogue field to evaluate
                processed[split] = data

        return DatasetDict(processed)

    def _populate_dataset_info(self) -> None:
        """
        Populate small, cheap-to-compute metadata for each split and store it in `self.dataset_info['splits']`.

        This function deliberately avoids computing heavy statistics (like per-example word counts).
        Those are computed lazily by `_get_story_stats` / `_get_dialogue_stats` when requested.
        """
        first_columns_set = False
        for split, data in self.dataset.items():
            # store per-split sample counts only; columns are expected to be identical across splits
            self.dataset_info['splits'][split] = {'num_samples': len(data)}
            if not first_columns_set:
                self.dataset_info['columns'] = list(data.column_names)
                first_columns_set = True

        # initialize caches for lazy stats
        self._story_stats_cache: dict[str, dict] = {}
        self._dialogue_stats_cache: dict[str, dict] = {}

    def _get_story_stats(self, split: str) -> dict | None:
        """
        Compute (and cache) min/max story word counts for a split.

        Returns:
            dict: A dictionary with 'min' and 'max' keys for word counts, or None if not applicable.
            None if the split does not exist or does not contain 'narrative' feature.
        """
        if split in self._story_stats_cache:
            return self._story_stats_cache[split]
        if split not in self.dataset:
            return None
        data = self.dataset[split]
        if 'narrative' not in data.column_names:
            return None

        min_c = None
        max_c = None
        # stream batches to avoid building a large list in memory
        for batch in data.iter(batch_size=1000):
            narratives = batch.get('narrative', [])
            counts = [len(n.split()) if n else 0 for n in narratives]
            if counts:
                bmin, bmax = min(counts), max(counts)
                min_c = bmin if min_c is None else min(min_c, bmin)
                max_c = bmax if max_c is None else max(max_c, bmax)

        stats = {'min': min_c, 'max': max_c}
        self._story_stats_cache[split] = stats
        return stats

    def _get_dialogue_stats(self, split: str) -> dict | None:
        """Compute (and cache) min/max dialogue word counts for a split. Returns dict or None if not applicable."""
        if split in self._dialogue_stats_cache:
            return self._dialogue_stats_cache[split]
        if split not in self.dataset:
            return None
        data = self.dataset[split]
        if 'dialogue' not in data.column_names:
            return None

        min_c = None
        max_c = None
        for batch in data.iter(batch_size=1000):
            dialogues = batch.get('dialogue', [])
            counts = []
            for entry in dialogues:
                if isinstance(entry, str):
                    counts.append(len(entry.split()))
                else:
                    counts.append(len(" ".join(entry).split()))
            if counts:
                bmin, bmax = min(counts), max(counts)
                min_c = bmin if min_c is None else min(min_c, bmin)
                max_c = bmax if max_c is None else max(max_c, bmax)

        stats = {'min': min_c, 'max': max_c}
        self._dialogue_stats_cache[split] = stats
        return stats

    def get_dataset_info(self, flat: bool = True, include_word_counts: bool = False) -> dict:
        """
        Return dataset_info as a single-level (flat) dictionary suitable for logging.

        Args:
            flat (bool): If True, return a flattened dict where nested keys are joined with dots.
                         If False, return the original nested `self.dataset_info` dict.

        Returns:
            dict: The dataset info dictionary (flattened if requested).
        """
        if not flat:
            return self.dataset_info

        flat_info: dict = {}
        # params: place keys at top level (no 'params.' prefix)
        params = self.dataset_info.get('params', {})
        for k, v in params.items():
            flat_info[k] = v

        # columns (single entry) - return as a list of strings
        cols = self.dataset_info.get('columns')
        if cols is not None:
            flat_info['columns'] = list(map(str, cols))

        # per-split sample counts (keys: num_samples.<split>)
        splits = self.dataset_info.get('splits', {})
        for split_name, split_info in splits.items():
            # use prefix 'split_name/' for keys, e.g. 'train/num_samples'
            if isinstance(split_info, dict) and 'num_samples' in split_info:
                flat_info[f"{split_name}/num_samples"] = split_info['num_samples']
            else:
                flat_info[f"{split_name}/num_samples"] = split_info

        # optionally include min/max word counts (computed lazily)
        if include_word_counts:
            for split_name in splits.keys():
                # story
                s = self._get_story_stats(split_name)
                if s is not None:
                    flat_info[f"{split_name}/story_min"] = s.get('min')
                    flat_info[f"{split_name}/story_max"] = s.get('max')
                # dialogue
                d = self._get_dialogue_stats(split_name)
                if d is not None:
                    flat_info[f"{split_name}/dialogue_min"] = d.get('min')
                    flat_info[f"{split_name}/dialogue_max"] = d.get('max')

        return flat_info

    def show_dataset_info(self, show_features: bool = False, show_word_counts: bool = False, show_dataset_info_details: bool = False) -> None:
        """
        Displays information about the loaded dataset.

        Args:
            show_features (bool): If `True`, displays the features of each split. Default is `False`.
            show_word_counts (bool): If `True`, computes and displays the minimum and maximum
                word counts for `narrative` and `dialogue` features in each split. Default is `False`.
        """
        if show_dataset_info_details:
            flat = self.get_dataset_info(flat=True)
            print("Dataset info (flattened):")
            print(json.dumps(flat, indent=2))
            print("-" * 40)

        for split, data in self.dataset.items():
            print(f"Split: {split}")
            # print per-split sample count using 'split_name/num_samples' format
            print(f"{split}/num_samples: {len(data)}")
            if show_features:
                print(f"Features: {data.column_names}")
            if show_word_counts:
                # use lazy cached stats to avoid remapping the dataset repeatedly
                if 'narrative' in data.column_names:
                    s_stats = self._get_story_stats(split)
                    if s_stats is not None:
                        print(f"Minimum story word count: {s_stats['min']}")
                        print(f"Maximum story word count: {s_stats['max']}")
                if 'dialogue' in data.column_names:
                    d_stats = self._get_dialogue_stats(split)
                    if d_stats is not None:
                        print(f"Minimum dialogue word count: {d_stats['min']}")
                        print(f"Maximum dialogue word count: {d_stats['max']}")
            print("-" * 40)

    def suffle_dataset(self, seed: int = 42) -> None:
        """
        Shuffles the dataset in place.

        Args:
            seed (int): Random seed for shuffling. Default is 42.
        """
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].shuffle(seed=seed)
            # reset caches since shuffling changes the order of examples
            self._story_stats_cache.clear()
            self._dialogue_stats_cache.clear()

    def duplicate_eod_examples(self, min_dupl: int = 3, max_dupl: int = 5) -> None:
        """
        Duplicates examples in the dataset for the train split that end with the dialogue end token.

        Args:
            min_dupl (int): Minimum number of times to duplicate each example. Default is 3.
            max_dupl (int): Maximum number of times to duplicate each example. Default is 5.
        """
        if self.dataset_info['params']['use_eos_as_eod']:
            raise ValueError("Cannot duplicate EOD examples when use_eos_as_eod is True.")
        if self.dataset_info['params']['data_types'] is None or 'train' not in self.dataset_info['params']['data_types']:
            raise ValueError("Train split must be loaded to duplicate EOD examples.")

        split = 'train'
        if split not in self.dataset:
            raise ValueError("Train split not found in dataset.")
        
        self.dataset_info['params'].update({
            'min_eod_duplication': min_dupl,
            'max_eod_duplication': max_dupl
        })

        def duplicate_eod_examples(batch):
            duplicated_batch = {key: [] for key in batch.keys()}
            for i in range(len(batch['dialogue'])):
                example = {key: batch[key][i] for key in batch.keys()}
                if example['dialogue'].endswith(DIALOGUE_END_TOKEN):
                    num_duplicates = random.randint(min_dupl, max_dupl)
                else:
                    num_duplicates = 1
                for _ in range(num_duplicates):
                    for key in batch.keys():
                        duplicated_batch[key].append(example[key])
            return duplicated_batch
        self.dataset[split] = self.dataset[split].map(duplicate_eod_examples, batched=True, desc="Duplicating EOD examples in train split")
        # reset caches since duplication changes the dataset
        self._story_stats_cache.clear()
        self._dialogue_stats_cache.clear()

    def set_tokenizer(self, tokenizer: AutoTokenizer) -> None:
        """
        Sets the tokenizer for the data loader.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to set.
        """
        self.tokenizer = tokenizer

    def set_chat_template(self, chat_template: str = "llama-3") -> None:
        """
        Sets the chat template for the data loader.

        Args:
            chat_template (str): The chat template to set.
        """
        self.chat_template = chat_template
        self.dataset_info['params']['chat_template'] = chat_template

    def formatting_prompts(self, examples: Dataset) -> dict:
        """
        Applies LLaMA-style chat formatting to the dataset. Not used on test dataset.

        Args:
            examples (Dataset): The dataset examples to format.

        Returns:
            dict: A dictionary containing the formatted texts, with key 'text'.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to apply LLaMA chat formatting.")

        texts = [] # Store formatted texts
        
        for narrative, dialogue in zip(examples['narrative'], examples['dialogue']):
            text = self.tokenizer.apply_chat_template(
                get_chat_template(narrative, dialogue, keep_assistance=True),
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        return {"text": texts}
    
    def formatting_prompts_test(self, examples: Dataset) -> dict:
        """
        Applies LLaMA-style chat formatting to the dataset. Only used on test dataset.

        Args:
            examples (Dataset): The dataset examples to format.

        Returns:
            dict: A dictionary containing the formatted texts, with key 'text'.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to apply LLaMA chat formatting.")

        texts = [] # Store formatted texts
        
        for narrative, dialogue in zip(examples['narrative'], examples['dialogue']):
            text = self.tokenizer.apply_chat_template(
                get_chat_template(narrative, dialogue, keep_assistance=False),
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        return {"text": texts}
