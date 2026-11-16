from abc import abstractmethod, ABCMeta

from src.fcga.selector import get_selector
from pathlib import Path


class BaseFunctionAugmenter(metaclass=ABCMeta):
    _model_dependent_criteria = {"gnnexplainer", "pgmexplainer", "subgraphx", "shapley", "output_drop"}

    def __init__(self, cfg):

        self.ratio = cfg.train.augmentation_ratio
        self.cfg = cfg
        self.node_budget = cfg.augmentation.params.node_budget
        self.epoch = None

        # get node idxs to augmented embeddings
        self.selector = get_selector(self.cfg)

        if self.cfg.dataset.name.startswith("MalNetTinyFeatures"):
            self.data_root_dir = (
                Path(self.cfg.injected.base_dir)
                / "datasets"
                / "augmented"
                / "mntf"
                / self.cfg.dataset.name
                / self.cfg.dataset.params.variant
            )

        else:
            raise NotImplementedError(f"Dataset not supported for augmentation: {self.cfg.dataset.name}")


    @abstractmethod
    def regenerate_augmented_embeddings(self):
        '''
        Regenerate augmented embeddings for all graphs. This is useful when we
        want to regenerate augmentation after some epochs, e.g. AT-like.
        '''
        # implicitly handle the case where graph_idxs are not populated.
        assert 'augmented_cache' in self.__dict__, \
            "Augmented cache not allocated. Please call propagate_information_from_dataset() before regenerate_augmented_embeddings()."

        if self.node_budget != -1:        
            assert self.topk_nodes is not None, \
                "Top-k nodes not populated. Please call populate_topk() before regenerate_augmented_embeddings()."



    @abstractmethod
    def propagate_information_from_dataset(self, dataset):
        '''
        Propagate necessary information from the dataset being augmented to the augmenter.
        As we apply an augmentation to a dataset, we assume that the dataset is sliced
            and we will augment everything in the sliced dataset.
        Shared memory allocation must be done here -- hence the abstractmethod tag.
        '''
        # these are for allocating shared cache for augmented embeddings
        self.graph_idxs = dataset.indices()
        self.dup_idxs = dataset.dup_idxs
        # this is needed for topk node selection, as the unique_graph_idxs
        # changes from the original dataset to the sliced dataset
        self.unique_graph_idxs = sorted(set(self.graph_idxs))
        # get feature dimension before possible LDP edits
        self.feature_dim = dataset._data.x.shape[1]


    def _lazy_load_topk(self):
        """Called by workers that find topk_nodes=None after augmentation has started.
        Only valid for model-dependent criteria whose selection runs after workers are forked."""
        criteria = self.cfg.augmentation.params.criteria
        if criteria not in self._model_dependent_criteria:
            raise RuntimeError(
                f"BUG: topk_nodes is None for non-model-dependent criteria '{criteria}'!"
            )
        past_epochs = [e for e in self.cfg.train.augmentation_start_epoch if e <= self.epoch]
        if not past_epochs:
            raise RuntimeError(
                f"Worker called _lazy_load_topk at epoch {self.epoch} before any augmentation_start_epoch was reached"
            )
        self.populate_topk(self._get_persistent_cache_dir(max(past_epochs)), refresh_selector=False)
        if self.topk_nodes is None:
            raise RuntimeError(
                f"Worker could not load topk_nodes from cache for criteria '{criteria}'; "
                "ensure the main process has run populate_topk before starting batch iteration"
            )

    def _get_persistent_cache_dir(self, epoch):
        '''Return the on-disk cache directory for selections at the given epoch.'''
        criteria = self.cfg.augmentation.params.criteria
        if criteria in self._model_dependent_criteria:
            base_checkpoint = getattr(self.cfg.train.params, 'base_checkpoint', None)
            if base_checkpoint is not None:
                # finetuning: share cache across all finetune runs from the same checkpoint
                return Path(base_checkpoint).parent / "selection"
            else:
                # standard training: epoch-scoped under save_dir (criteria appended by selector)
                return Path(self.cfg.injected.save_dir) / "selection" / str(epoch)
        elif criteria in {"degree", "in_degree", "out_degree", "pagerank", "random", "all", "betweenness", "closeness", "ppr"}:
            return self.data_root_dir / "selection"
        else:
            assert False, f"Unexpected criteria with cache path: {criteria}"


    def populate_topk(self, cache_dir, refresh_selector=False):
        '''
        Load/generate topk nodes for augmentation.
        If refresh_selector is False, tries cache only and sets topk_nodes to None if missing.

        Note that because graph_idxs are global, it needs to take in the original dataset (i.e. injected), and
            not the one being augmented (i.e. the sliced one which we get from propagate_information_from_dataset),
            as the sliced one has different graph indices.
        '''
        cache_dir = Path(cache_dir)
        if refresh_selector:
            self._selecting = True
            self.topk_nodes = self.selector(
                self.cfg.injected.dataset,
                graph_idxs=self.unique_graph_idxs,
                cache_dir=cache_dir,
            )
            self._selecting = False
        else:
            self.topk_nodes = self.selector(
                self.cfg.injected.dataset,
                graph_idxs=self.unique_graph_idxs,
                cache_dir=cache_dir,
                check_cache_only=True,
            )


    @abstractmethod
    def __call__(self, batch):
        epoch = int((self.cfg.injected.temp_dir / "epoch.txt").read_text())
        self.epoch = epoch
    

    def set_epoch(self, epoch):
        self.epoch = epoch
        (self.cfg.injected.temp_dir / "epoch.txt").write_text(str(epoch))

        if self.epoch in self.cfg.train.augmentation_start_epoch:
            self.populate_topk(self._get_persistent_cache_dir(self.epoch), refresh_selector=True)
            self.regenerate_augmented_embeddings()

        # in case first load, load topk nodes from cache without regenerating
        elif not hasattr(self, 'topk_nodes'):
            if self.epoch < min(self.cfg.train.augmentation_start_epoch):
                # __call__ returns early before the lazy-load check anyway; skip expensive generation
                self.topk_nodes = None
            else:
                past_epochs = [e for e in self.cfg.train.augmentation_start_epoch if e <= self.epoch]
                if past_epochs:
                    self.populate_topk(self._get_persistent_cache_dir(max(past_epochs)), refresh_selector=False)
                if self.topk_nodes is None:
                    self.populate_topk(self._get_persistent_cache_dir(self.epoch), refresh_selector=True)
                self.regenerate_augmented_embeddings()
