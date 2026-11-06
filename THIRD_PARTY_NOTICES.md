# Third-party notices

**Built with PriorLabs-TabPFN**

TCAP uses the [TabICL](https://github.com/soda-inria/tabicl) and [TabPFN](https://github.com/PriorLabs/TabPFN) model libraries.

- `src/tcap/backends/_tabicl_finetune/` contains classification fine-tuning support derived from TabICL's `src/tabicl/_finetune/`. Its upstream copyright and license terms are retained in [licenses/TabICL-LICENSE](licenses/TabICL-LICENSE).
- `src/tcap/backends/_tabpfn_finetune/` contains classification fine-tuning support derived from TabPFN's `src/tabpfn/finetuning/`. Its copyright headers are retained. The applicable Prior Labs License is included in [licenses/TabPFN-LICENSE](licenses/TabPFN-LICENSE).

Internal fine-tuning imports are relocated to the `tcap.backends` namespace. The TabPFN training support includes estimator-view gradient microbatching. Upstream licenses apply to their respective derived files.
