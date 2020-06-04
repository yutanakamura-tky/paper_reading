###### 医療言語処理のためのデータセットと言語資源
- データセット
    - 語彙正規化 (Entity linking, entity normalization)
        - The NCBI disease corpus
    - 診療記録
        - Cleveland Clinic EHRs (ClvC)
        - i2b2 EHRs (i2b2)
    - 文献や教科書など
        - Medical Literature (MedLit)
            - 教科書, ガイドライン, Wikipedia
            - 具体的には Wikipedia, DynaMed, Elsevier, Wiley
- 言語資源
    - メタシソーラス
        - UMLS
            - MetaMap
                - MBR: MEDLINE/PUBMED Baseline Repository
                - MMO: MetaMap Machine Output
            - CUI (Unique Concept Identifier)
    - UMLSに所属する種々のオントロジー
        - MeSH
        - SNOMED-CT
    - オントロジー(UMLSに所属しているかどうか不明)
        - Gene Ontology (GO)
        - the Chemical Entities of Biological Interest (ChEBI) ontology 
        - the Human Phenotype Ontology
        - Disease Ontology (DO)


###### Dataset / Shared task
- [BioNLP-OST 2019](https://2019.bionlp-ost.org/tasks)
    - CRAFT task: Integrated structure,  semantics and coreference task
    - PharmaCoNER task: Pharmacological substance, compound and protein named entity recognition 
    - AGAC task: Entity/relation recognition and GOF/LOF mutated gene text identification task based on the Active Gene Annotation Corpus
    - Bacteria-Biotope Task: Extracting information about bacterial biotopes and phenotypes
    - Seedev Task: Event extraction of genetic and molecular mechanisms involved in plant seed development
    - RDoc Task: Information Retrieval and Extraction on Mental Health using Research Domain Criteria

- NLI
    - MedNLI
- NER
    - 疾患
        - BC5CDR
        - NCBI-disease
        - Variome
    - Gene/Protein
        - BC2GM
        - CRAFT
        - JNLPBA
- 質問応答
    - EMNLP 2019
        - [PubMedQA](https://www.aclweb.org/anthology/D19-1259/)
            - 質問に対し, PubMed文献を利用して yes/no/maybe で返答し, かつその根拠となる区間を提示するタスク.
    - NAACL 2018
        - [CliCR](https://arxiv.org/abs/1803.09720)
            - 医療言語処理の機械読解データセット. ソース文は症例報告.
            - 問題文の1単語が空白となっており, 当てはまる単語をソース文をもとに正しく推定する.
    - BioASQ, 2019
    - BioASQ, 2015
    - emrQA, 2018
    - BioRead, 2018
        - 論文を対象にしたQA dataset.
    - BMKC, 2018
        - 論文を対象にしたQA dataset.
    - QA4MRE, 2013
        - Alzheimer病に関するQAデータセット
- Temporal Extraction
    - THYME corpus
- 知識獲得 Taxonomy Learning
    - [SemEval 2018 Task 9](https://competitions.codalab.org/competitions/17119#learn_the_details-terms_and_conditions)


###### テンプレ
- なに
- これまでと比べてどうすごい
- 技術の肝
- 検証
- 議論
- 次に読むべきもの



###### Radiology Gamuts Ontology: Differential Diagnosis for the Semantic Web. (Radiographics 2014)

[https://pubs.rsna.org/doi/full/10.1148/rg.341135036](https://pubs.rsna.org/doi/full/10.1148/rg.341135036)


###### Structural Scaffolds for Citation Intent Classification in Scientific Publications. (NAACL 2019)

- SciCiteを紹介した論文.

https://www.aclweb.org/anthology/N19-1361/


###### Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks. (ACL/ICJNLP 2015)

https://www.aclweb.org/anthology/P15-1150/

###### Biomedical Event Extraction based on Knowledge-driven Tree-LSTM. (NAACL 2019)

https://www.aclweb.org/anthology/N19-1145

###### Visualizing and Understanding the Effectiveness of BERT. (EMNLP/ICJNLP 2019)

https://www.aclweb.org/anthology/D19-1424
https://www.google.com/url?q=https%3A%2F%2Fspeakerdeck.com%2Fyummydum%2Fsnlp-presentation-20190925&sa=D&sntz=1&usg=AFQjCNHhmYKnVgFMURoebJoIoAC9c3BIAg


###### Qian Chen et al. Enhanced LSTM for Natural Language Inference. ACL 2017.

https://www.aclweb.org/anthology/P17-1152


###### Soumya S et al. Incorporating Domain Knowlegde into Medical NLI using Knowledge Graphs. EMNLP 2019.

- どんなもの?
    - 生物医学領域におけるNLIに知識を導入しようとした研究の1つ.
- 先行研究と比べて何がすごい?
    - 先行研究:
        - Chen et al., 2018
        - Romanov and Shivade 2018
        - Lu et al., 2019
        - Jin et al., 2019
- 技術のキモは?
    - ESIM model
- どうやって検証した?
- 議論はある?
- 次に読むべきものは?

https://www.aclweb.org/anthology/D19-1631/

###### 68. Membership Inference Attacks Against Machine Learning Models (2017)
- なに
    - 多クラス分類モデルをターゲットとした membership inference.
- 先行研究とくらべてどうすごい
    - membership inferenceをはじめて提唱した研究と思われる
- 手法
    - Target modelに実際の学習データが入力されたときの出力はhigh confidenceになるだろうという仮定にもとづく.
    - まずデータ空間Sを探索し, target modelが高い確信度でクラスCkに分類するような部分空間Skを探す.
    - 次に, Skからの標本をx(in)k, x(in)kに対するtarget modelの出力をy(in)kとし, (x(in)k, y(in)k)に"in"のラベルを貼る.
    - さらにSkと離れた部分空間からの標本をx(out)k, x(out)kに対するtarget modelの出力をy(out)kとし, (x(out)k, y(out)k)に"out"のラベルを貼る.
    - 上記をもとに, データ点およびtarget modelの出力値からin/outを推定する2クラス分類を解くモデルを構築し, これをattackerとする.
- 検証
    - 定量評価はattackerがtarget modelのtraining setに入っているデータとtest setに入っているデータを見分ける性能によって行なった.
    - 各種データセットで検証. 画像とtableデータ. tableデータセットのうち1つは医療情報(Texas hospital stays).
    - attackerは高いaccuracyとprecisionを達成した.
    - また, attackerに本来のtarget modelのタスクを解かせるとかなり性能が悪く, attackerをtarget modelの学習データに強くoverfitさせることに成功していることも判明した.
- 議論
    - 多クラス分類モデルでさえあればこの方法でなんでも攻撃できてしまう.
    
https://ieeexplore.ieee.org/document/7958568


###### 67. Pre-training of Graph Augmented Transformers for Medication Recommendation (IJCAI 2019)
- なに
    - G-BERTを提案した論文.
    - 言語データではなく, 診断コード(ICD-9)と薬剤コード(ACT)のオントロジー内における位置をembedding化しようとしている.
- これまでと比べてどうすごい
    - BERTをontology embeddingの構成に利用した先行研究はない
- 技術の肝
    - 1. 問題設定
        - 患者がこれまでにM-1回入院しているとする
        - t回目の入院時につけられた診断のICD-9コードの集合をC_d^(t)とする
        - t回目の入院時に使用された薬剤のICD-9コードの集合をC_m^(t)とする
        - 目的は, M回目の診断が与えられたとき, M回目の薬剤を予測することで, drug recommendationをすること
        - つまり C_d^(1),...,C_d^(M),C_m^(1),...,C_m^(M-1) を入力として C_m^(M) を正しく予測したい
    - 2. 全体像
        - C_d^(t)に属する1つの疾患コードを c_d^(t) とおく
        - すべての c_d^(t) に対してICD-9 graph embedding を利用した疾患特徴量 o_d^(t)を得て, それをconcatする
        - concatしたものをBERTに入力して visit embedding v_d^(t)を得る
        - C_m^(t)に属する1つの疾患コードを c_m^(t) とおく
        - すべての c_m^(t) に対してACT graph embedding を利用した薬剤特徴量 o_m^(t)を得て, それをconcatする
        - concatしたものをBERTに入力して visit embedding v_m^(t)を得る
        - t=1,...,M-1についての v_d^(t) の平均, v_m^(t) の平均を求め, それをconcatし, さらにそれと v_d^(M) をconcatする
        - softmaxに通して C_m^(M) を予測する
    - 3. Graph embeddingの学習
        - 大雑把にいうと
            - c_d^(t)のノード特徴量, c_d^(t)の子ノードの特徴量, initial embeddingを用いて, c_d^(t)のenhanced embedding (h_d^(t)とする)を更新する
            - c_d^(t)の親ノード特徴量, c_d^(t)の親ノードの特徴量, enhanced embeddingを用いて, 疾患特徴量o_d^(t)を得る
            - o_d^(t)からloss functionが計算できるので, 誤差逆伝搬により各ノードの特徴量が更新される
            - 薬剤についても同様のプロセスを行う
            - enhanced embedding, 疾患/薬剤特徴量を得るための関数の一部にAttentionを利用する(Graph Attention Network)
    - 4. BERTの利用
        - BERTの1トークンのかわりとなるのは1疾患特徴量もしくは1薬剤特徴量
        - 患者がその入院回で5疾患が診断されていればシーケンス長も5となる
        - シーケンスの順序に意味はないのでposition embedding機構は取り除いてある
        - BERTの重みは疾患と薬剤でshareする
    - 5. 事前学習
        - 
- 検証
    - MIMIC-IIIのデータセットで検証し, 既存手法よりAUC, Jaccard係数, F1 scoreがいずれも改善
- 議論
- 次に読むべきもの



###### 66.1 Big Data in Public Health: Terminology, Machine Learning, and Privacy (2018)
- なに
    - 公衆衛生におけるビッグデータの影響について述べたレビュー論文.
- 次に読むべきもの
    - EHRが善意の誤解によって誤って公開されてしまった事例(ref.80, 93)
    - データの次元数が上がることによって匿名化データから個人が演繹的に特定可能となる可能性(ref.110, 125)


###### 66. MEDTYPE: Improving Medical Entity Linking with Semantic Type Prediction (2020)

- なに
    - 生物医学領域におけるeneity linking (ここではentityをUMLSに正しく標準化すること) の性能を, semantic typeの分類タスクを間に挟むことによって改善させた論文
- これまでと比べてどうすごい
    - これまで semantic type は十分に活用されてこなかった
- 技術の肝
    - MEDTYPE
        - NERでentityを抽出
        - 次に, そのentityと周囲のcontextをBERTに入力し, semantic typeを\[CLS\]の特徴量を利用して多クラス分類
        - こうすることでUMLSの正規化先のconceptの候補を減らせる
    - WikiMed
        - しかしそのための医学領域のアノテーション付き大規模データセットを用意するのは難しい
        - そこでWikipediaを利用したデータセットをまず構築した
        - もう少し正確にいうと, Wikipedia記事とUMLS entityを紐づけたデータセット
            - Wikidata
            - Freebase
            - NCBI Taxonomy
        - Wikipedia記事 -> UMLS entity が約6万組ある
        - 文 -> UMLS entity を65万組抽出した (Wikipediaの600万記事から)
    - PubMedDS
        - Distant supervisionを利用するためのデータセット
        - PubMed abstractから構成している
        - まずSOTAのNERモデルで, 各記事からentityを抽出
        - 次に, そのentityのうち記事のMeSHタグと完全一致するものだけをフィルターする
        - MeSHタグはUMLS上でentityとすでに紐付けされてあるので, そのmappingを利用する
- 検証
    - データセット
        - NCBI disease corpus
        - Bio CDR
        - ShARe
            - MIMIC-IIから作成したデータセット
        - MedMentions
            - 2019年に作成されたデータセット
    - 補助するentity linker
        - MetaMap
        - cTakes
        - MetaMapLite
        - QuickUMLS
        - ScispaCy
- 議論
- 次に読むべきもの
    - [Distant Supervisionの医学領域への利用. ACL 2018](https://www.aclweb.org/anthology/W18-3026/)
    - [医療文書を容易に持ち出してはいけないことについて](https://pubmed.ncbi.nlm.nih.gov/12234714/)

https://arxiv.org/pdf/2005.00460v1.pdf


###### 65. ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models (2018)
- なに
    - BERTの入力に摂動を与えることでdownstream taskに対する性能を大幅に低下させることに成功した
- これまでと比べてどうすごい
    - 従来手法では, BERTの入力に与える摂動はhuman crafted featureに頼っており, 文脈も考慮できない
    - 本手法では, 摂動を別のBERTモデルによって決定することで文脈を考慮した摂動を与えることに成功
    - つまり入力文の文法的/意味的な自然さを保ったままBERTの推測を誤らせることに成功している
    - さらに本論文の手法はLSTMに対しても有効であった
- 技術の肝
    - まず攻撃用BERTモデルを用意する
    - 1. 脆弱な単語の検出
        - ターゲットBERTが入力文のうちどの単語にもっとも依存しているかを検出する
        - これはターゲットBERTに, もとの入力文Sのどれか1単語を[MASK]して入力することを繰り返すことで判別可能
        - ターゲットBERTの出力の変化量が大きい順ほど脆弱な単語であると考える
        - こうして脆弱な単語の候補 {w0, w1, ..., } が得られる
    - 2. 摂動の与え方
        - 対象とした脆弱な単語 wi が高頻度語(=そのままでvocabに登場する)か低頻度語(=subwordsの形でしかvocabに登場しない)かで異なる
        - 高頻度語の場合
            - 工程1で, wi をMASKした場合のMasked Language Modelの予測結果が得られているはず
            - そこでその予測結果から上位K単語 (wi1, wi2, ..., wiK) をえらぶ
            - NLTK辞書を参照し, wiと対義語になるものを除く
            - Sのwiをwij (j=1,...,K) に置換した文をターゲットBERTに与え, ターゲットBERTの予測が誤った段階で攻撃を終了する
        - 低頻度語の場合
            - wi が M個の subword (z1,...,zM) からなっているとする
            - 工程1で, wi をMASKした場合のMasked Language Modelの予測結果が得られているはず
            - そこでその予測結果から, 摂動の候補となる長さMのトークン列を上位K個得る (wi11,..,wi1M,...,wiK1,...,wiKM)
            - ただしトークン列のうち全体が1つのwordに復元できないようなものは除く
            - Sのwiをwij1,...,wijMに相当する単語 (j=1,...,K) に置換した文をターゲットBERTに与え, ターゲットBERTの予測が誤った段階で攻撃を終了する
- 検証
    - 攻撃の有効性
        - 文書分類タスク, NLIタスクで有効性を検証
        - いずれのタスクでも攻撃によってaccuracyを10%未満にまで落とすことに成功した
    - 摂動を与えたあとの文の自然さ
        - 人手で評価した
        - もとの文と, 摂動を与えた文のそれぞれについて, 対象のNLPタスクと同じタスクを行わせ, さらに意味的/文法的な自然さも評価させた
        - その結果, 摂動を与えたあとの文は人間にとって意味的に自然であり, かつNLPタスクのサンプルとしても同等であることが示された
- 議論
- 次に読むべきもの
https://arxiv.org/abs/1806.01246



###### 64. The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks. (2018)
- なに
    - 言語モデルが意図せずに個人情報を暴露してしまう傾向に対する評価指標と, その評価方法を導入した論文.
        - たとえば "私の口座番号は, " と入力してそのオートコンプリート結果を参照したりすることによって個人情報が漏洩する
        - このような "意図しない記憶" は, 対策がとられない限り起きてしまう
        - たとえば PTBツリーバンクデータセットに "私の社会保障番号は078-051120です" という文を追加してLSTMを訓練し, "私の社会保障番号は" のつづきを貪欲法によって推定すると, 社会保障番号が正しく復元できてしまう
        - 
    - モデルが文を入力されるとlogitを返すような状況を想定している.
    - このためにはモデル学習時に, 学習データに無作為に選んだ"カナリア文字列"を混ぜておく. 詳細は後述
- これまでと比べてどうすごい
- 技術の肝
    - "意図しない記憶" を定量評価する方法
        - カナリア文字列の作り方
            - 文字列の一部をランダムにフォーマットした文字列を考える
            - たとえば f'私の社会保障番号は{ランダムに選ばれた9桁の数値}です' など
            - カナリア文字列をどうつくるかは, 定量評価にあまり影響しないことが実験でわかっている
            - カナリア文字列はコーパスに登場しない類のものがよい
                - コーパスによく登場する文面からカナリア文字列を作ると, 本来の目的に反する場合がある
                - 理由はモデルが文字列を丸暗記することのすべてが悪いとは限らないため
                    - 特定のドメインに役立つような言い回しを丸ごと記憶している場合, その丸暗記は悪いものとはいえない
                    - 逆に, モデルの目的に役立たない文字列であるにもかかわらず個人情報を含む文字列を丸暗記している場合は悪い丸暗記だといえる
        - カナリア文字列のrank
            - 可能なカナリア文字列集合Sの要素sのrankとは, あるモデルMにSの全要素のパープレキシティを計算させたとき, sのものはその中で何番目に小さいかという順位
        - モデルのexposure
            - モデルが存在することによって, カナリア文字列がどのくらい正しく当てやすくなったかという指標
            - モデルがなければ, 全く当てずっぽうで推測するしかないので, エントロピーは集合Sの大きさの1/2
            - モデルがあれば, rankの高い順にカナリア文字列かどうかを確認するものとすれば, エントロピーはカナリア文字列のrankと等しい
            - よって, exposure = log_2(|S|) - log_2(rank(s)) と定義する
        - より効率的なexposureの計算
            - 非対称正規分布ρ(x)によって exposure = - log_2 (\int_{0}^{Perplexity(s)} ρ(x)dx) と近似する
- 検証
    - 低いパープレキシティの文を当たる方法との関連性
        - 総当たり方はあまりにも非効率なので, ダイクストラ法によって低パープレキシティの文の候補を得る
        - exposure > 30 のカナリア文字列に対しては, ダイクストラ法によって候補を100程度まで減らせた
- 議論
    - モデルからの個人情報にも色々あるが, その中でも "意図しない記憶" にのみ焦点をあてた研究である
- 次に読むべきもの


###### 63. Algorithms that remember: model inversion attacks and data protection law (2018)
- 機械学習とEU一般データ保護規則(GDPR)に関する関わりと, 機械学習モデルに対するinversion attackについて概説した論文.
- GDPRには次のことが規定されている:
    - 同意, 契約, その他の法的根拠がなければ個人データを用いてモデルを訓練することはできない
    - データ提供者にはモデルを訓練する目的が知らされるべき
    - データ提供者には同意を撤回する権利が与えられるべき
    - モデルを訓練する者は, データ提供者の求めがあればモデルの決定ロジックなどについて説明しなければならない
- モデルに対する攻撃には大きく2種類ある
    - (1) Model Inversion
        - 攻撃者はあるaという人の個人情報x_aを知っている. 攻撃者はx_aを利用して, モデルMの学習に使用されたaの別の個人情報y_aを復元する.
    - (2) Membership Inference
        - 攻撃者はあるaという人の個人情報x_aを知っている. 攻撃者はx_aを利用して, モデルMの学習にaの個人情報が使われたかどうかを推測する.
- モデル自体が個人情報とみなされる判例がある
- モデルが個人情報と本格的に扱われるようになった場合に発生しうる権利
    - データ提供者が自身のデータをモデルの学習データから除外するよう求める権利
    - データ提供者がモデルを使わない権利
    - データ提供者がモデルの出所や, 誰に使われたり送信されたりしているかを知る権利
    - しかしこれらは, データ自体が匿名化されることが多いため権利としては行使しにくいかもしれない (それに, モデル作成者もまた, データ提供者の連絡先などの情報はもはや持っていないことが多い)
    - また, データ提供者が自身をモデルに忘れ去ってもらいたいと思っても, 技術的に難しいことが多い
        - 技術的理由には再学習のコストなどもふくむ
- モデルが個人情報と本格的に扱われるようになった場合に発生する義務
    - モデル作成者はモデルがそもそも個人情報を漏洩していないことを保証しなければならない (GDPR第25条(1))
    - また, EU圏外にモデルが転送される場合には扱いはより厳しくなるものと思われる
    - モデル作成者は一定以上古いデータを削除する義務がある? (保持制限)
- モデルへの攻撃に対する対処法
    - 差分プライバシー
    - データセットに含まれるすべての個人について、その個人をデータセットから削除しても、そのクエリの結果が目立って変わらないように、クエリに正確な形式のランダムノイズを追加するという概念
    - しかし差分プライバシーは計算コストがかかるうえ, モデルの性能を保ちながら実現させることが難しく, 機械学習モデルに対して実現できるプレイヤーは限られている



https://royalsocietypublishing.org/doi/full/10.1098/rsta.2018.0083


###### 45. MIMIC-III or NER shared task 強化期間: EMNLP 2019
- BioNLP-OST 2019
    - 

###### 48. PubMedQA: A Dataset for Biomedical Research Question Answering (EMNLP 2019)
- なに
    - PubMedからresearch questionに応答する質問応答データセットを作成
    - 内容は
        - 1. 質問文: 原著論文のタイトルと同一, またはそれをもとにして作成
        - 2. context: abstractのうちconclusion以外の部分
        - 3. long answer: asbtractのconclusion
        - 4. answer: yes/no/maybeの3択
- これまでと比べてどうすごい
    - 既存のBiomedical domain QA datasetは, サンプル数が少ない, もしくはfactoidであり推論を必要としていない
- 技術の肝
    - 人工的にnoisy labeled datasetを作成してサンプル数を大きくした
    - 次の3つのデータセットを用意:
        - PQA-L: 原著論文のタイトルがすでに疑問文の形をしている. yes/no/maybeは人手でアノテーション
        - PQA-U: PQA-Lのyes/no/maybeラベルがないもの
        - PQA-A: 疑問文でない原著論文タイトルから疑問文を作成し, yes/noは自動でラベリング(noisy label)
            - 92.8%がyes labelで7.2%がno label (著しく偏っている)
    - ベースライン手法は以下のとおり:
        - 用語の定義
            - reasnoning-required: long answerを当てるタスクも並行して行う
                - question & contextの各BoWがlong answerに含まれているかどうかを2値判定するタスクも並行して行いbinary cross-entropy lossも使用する
            - reasnoning-free: long answerを当てるタスクは行わない(yes/no/maybeだけ当てる)
        - Multi-phase Fine-tuning
            - Phase I fine-tuning on PQA-A
                - BioBERTをfine-tuningする(θ0->θ1)
                - reasnoning-required settingにてPQA-Aを利用
            - Phase II fine-tuning on bootstrapped PQA-U
                - 以下はすべてreasoning-free setting
                    - PQA-AでBioBERTをfine-tuning(θ0->θB1)
                    - PQA-LでBioBERTをfine-tuning(θB1->θB2)
                    - PQA-Uをpseudo labelする
                - 次にreasoning-required settingにて...
                    - pseudo labelしたデータでBioBERTをfine-tuning(θ1->θ2)
                - Final Phase
                    - reasoning-required settingにてPQA-LでBioBERTをfine-tuning(θ2->θF)
- 検証
    - 上記のMulti-phase手法でaccuracy 68.08, F1 52.72
    - 人間2人の合議ではaccuracy 90.40, Macro-F1 84.18
    - 人間1人ではaccuracy 78.0, Macro-F1 72.2
- 議論
- 次に読むべきもの

https://www.aclweb.org/anthology/D19-1259/


###### 47. Open Sesame: Getting inside BERT's Linguistic Knowledge (ACL 2019)

- なに
    - BERTの各層がどのように構文などの情報を保持しているかを分析した論文.
- これまでと比べてどうすごい
    - これまではどのような研究はあまり存在しなかった
    - LSTMについてはEMNLP 2018にdiagnostic classifierという手法による先行研究がある
- 技術の肝
    - 検証1: diagnostic classifier
        - main auxiliary: 複文に複数出現する助動詞のうちどれが主文にかかっているものかを当てるタスク
        - noun:
            - compound: 複合名詞を構成する名詞のうちどちらがメインの名詞かを当てる
            - possessive: A' B という所有格の形になっている主語に対して B のほうが真の主語であることを当てる
        - n-th token: トークンが何番目の語であるかを当てる
    - 検証2: 照応解析
        - subject-verb: 動詞を主語にあわせて正しく活用させるタスク
        - reflexive anaphora: 再帰代名詞を文の内容に応じて正しく使い分けるタスク
- 検証
    - 検証1の結果
        - n-th token:
            - 予想どおり低レイヤーのattentionほど高成績になった
            - またpositional encodingのレイヤーを削除すると著明に成績が落ちた
        - main auxiliary:
            - 高レイヤーのattentionほど高成績になった
        - noun:
            - compoundのみ, BERT-base-uncasedでは高レイヤーほど高成績になった
            - しかしcompoundのBERT-large-uncasedは低成績. 理由は不明
            - またpossessiveはBERTのモデルの大きさによらず低成績. 理由は不明
- 議論
- 次に読むべきもの

https://www.aclweb.org/anthology/W19-4825

###### Leveraging Medical Literature for Section Prediction in Electronic Health Records (EMNLP 2019)
- #29ですでに紹介済みです...
- なに
    - 電子カルテの診療記録のどこからどこまでが何の情報であるかを正しく推定するタスク(section detection)の改善手法.
- これまでと比べてどうすごい
    - タスクの内容
        - 先行研究ではさまざま
            - section detection only
            - section classificaton only (with known section boundaries)
            - section detection + section classification
        - 本研究では section detection + section-level classification 
    - 手法
        - Section predictionの先行研究はhandcraft featureに依存した非ニューラル手法
        - 本研究では deep learning (特に転移学習)
    - データセット
        - 2つの先行研究が複数のデータセットで検証している
        - 最大のデータセットはi2b2 dataset
- 技術の肝
    - データセットは3つ
        - Medical Literature (MedLit)
        - Cleveland Clinic EHRs (ClvC)
        - i2b2 EHRs (i2b2)
    - 11 class分類した. これは先行研究通り
        - Allergies
        - Assessment and Plan
        - Chief Complaing
        - Examination
        - Family History
        - Diagnostic Findings
        - Medications
        - Past Medical History
        - Personal and Social History
        - Procedures
        - Review of Systems
- 検証
- 議論
- 次に読むべきもの


###### 44. Exploring Diachronic Changes of Biomedical Knowledge using Distributed Concept Representations (ACL 2019)

-どんなもの？  
    - 疾患の治療法や, 薬剤の適応疾患の変遷を可視化した研究.  
    - より大きな目的としては, 医学知識がどれだけ信頼できるかどうかをその新旧を考慮したうえで判定したいという目論見がある.  

-先行研究と比べてどこがすごい？ 
    -   

-技術や手法のキモはどこ？  
    - MEDLINE論文にMetaMapを適用した結果を利用 (MetaMapped 2015 MEDLINE Baseline Results) 
        - Title/AbstractのUMLSに該当する語がそれぞれのIDに置換されている  
    - MEDLINE文献を年代順に12グループに分け, それぞれのIDに対するembeddingをそれぞれFastTextで訓練した 
    - さまざまなUMLS IDの類似度 (cosine similarity) の年代による変遷を可視化してみた  

-どうやって有効だと検証した？  
    - 医学生が選んだUMLS entityの組についていくつか例を提示してある  
        - ミノキシジルと高血圧の類似度が下がり, ミノキシジルとhair lossの類似度が上がった  
        - Microprolactinomaと経篩骨下垂体切除術の類似度が下がり, 経蝶形骨下垂体切除術, D2アゴニストとの類似度が上がった  
        - 他にもCMLとイマチニブ, HCVと抗ウイルス薬など  

-議論はある？  
    - すでに明らかになっている知識を再確認しただけでは?  

-次に読むべき論文は？  


https://www.aclweb.org/anthology/W19-5037

###### 41. An efficient prototype method to identify and correct misspellings in clinical text (BMC Res Notes 2019)
- なに
- これまでと比べてどうすごい
- 技術の肝
    - 手法
        - 前処理
            - tokenize
            - stop word除去
            - 大文字のみ, 数字のみ, 4文字未満のトークンも除去
            - 残ったトークンはすべて小文字化
        - word2vec
            - 2種類のコーパスに対してそれぞれ独立に訓練した
            - CBOW, window=5, hidden_dim=500, epoch=10. 5回以上登場した単語のみを対象とした
            - 出現回数の上位1000トークンをtarget termsと定義
        - ベクトルとしての類似度が高く, かつスペルが誤っている語の選出
            - 各target termsに対し, ベクトルの類似度が高いトークンを上位1000語pick up
            - SPECIALIST Lexiconを用いて, スペルの正しいトークンを除去
            - さらに句読点を含むもの, 数字を含むもの, 4文字未満のものも除去
            - 残ったトークンをすべて小文字化
            - さらにLevenshtein距離が1から3のもののみを, そのtarget termsのpotential misspelling(variants)として取り上げた
- 検証
    - Veterans Health Administration (VHA) のデータを使用
        - 世界最大のintegrated health care system
            - 800万患者, 1243施設をカバー
        - Veterans Affairs Informatics and Computing Infrastructure (VINCI) というプラットフォームが研究者用に提供されている
        - 本研究で用いたのは...
            - 病理レポート50000件 (Surgical pathology notes (SP)
            - 26786救急外来診療記録 (Emergency Department Visit and Progress Notes (EDVP))
    - 検証
        - 3人のアノテーターを用意
            - target termsは正しいスペルの語となっているか? (True/False)
            - potential variantsは誤ったスペルの語となっているか? (True/False)
            - 意見が割れた場合は多数決とした
    - 結果
        - データ
            - EDVP corpus の 1000 target terms に対して 235 potential misspellings(vatiants)
            - SP corpus の 1000 target terms に対して 53 potential misspellings(variants)
            - EDVPのほうがスペルミスは多かった
                - EDVPのpotential variantsの45%は38 target terms由来
                - SPのpotential variantsの30%は7 target terms由来
        - 手法の結果
            - PrecisionはSPで0.9057, EDVPで0.8979
            - False positiveの例: スペルの正しい別の語, 別の語のスペルミス, slang equivalent, 意味をなさない語
        - 抽出されたスペルミスの特徴
            - 5種類
                - insertion
                - omission
                - transposition
                - wrong letter
                - mixed/multiple error type
            - 特徴
                - 上記5種類の分布はSPとEDVPで似ていた
- 議論
    - これはdetectionとcorrectionを兼ねている
    - 私の疑問
        - 同じ potential misspellings が異なる target terms に対して出現することはあったか?
        - Precisionは target terms単位で計算されているのか? それとも potential misspellings単位か?
        - Precisionは各 potential misspellings の出現頻度で重みづけされているか?
- 次に読むべきもの



###### 40. Medical Entity Linking using Triplet Network. (NAACL 2019)

- なに
    - 語彙正規化 (Entity Normalization) のための, hand-craft featureが不要な手法を提案した
    - Triplet Networkを使用している
- これまでと比べてどうすごい
    - Hand-craft featureが不要
- 技術の肝
    - 1段階目: Candidate generation
        - Knowledge Baseから病名のpotential candidateを作成
        - ここでは Knowlegde BaseとしてMEDIC Lexiconを利用
        - 目的の disease mention が l単語からなる場合, そのl単語分の word embedding の和をとり, v0とする
        - 次に, 各 Knowlegde Base に収録されている語についても, その語を構成する単語の word embedding の和をとり, v1, ..., v_n_kbとする
        - Step 1
            - v0とvi (i=1,...,n_kb) のコサイン類似度をとり, 閾値 t1 以上のもののうち上位 k1 個だけを候補として残す
        - Step 2
            - v0とvi∈{vj|1<=j<=n_kb, cos(v0,vj)>=t1} のJaccard係数をとり, 閾値の t2 以上のもののうち上位 k2 個だけを候補として残す
        - 文献では t1=0.7, t2=0.1, k1=3, k2=7
    - 2段階目: Candidate ranking
        - potential candidateをrankingする
        - Triplet Networkを利用する
            - まず訓練用データとして (m,p,ni)=(disease mention, pos candidate, i-th neg candidate)のtripletを作る
            - 次に m,p,ni の word embedding をそれぞれCNNに通して hm, hp, hni を得る
            - この CNN は重みを共有する
                - 1d conv + max-pooling + ReLU
            - 距離 dp=dis(hm,hp), dni=dis(hm,hni) を計算する
                - L2 distance
            - Loss = max(dp-dni+α, 0)  (αは正例と負例の距離の差を最低限どれだけ要求するか)
                - Adam lr=0.001, early stoppingを50epochごとに
    - 使用したWord embedding
        - Wikipedia, Pubmed PMC-corpus で訓練した200次元の word2vec
        - out-of-vocabularyな語彙については PubMed と MIMIC-III で訓練した 200次元のFaseText
            - Fasetextのwindow size=20, lr=0.05, sampling thresh=1e-4, negative example=10
- 検証
    - NCBI disease dataset で検証し, accuracy 90%
    - 既存手法の86.1%より大きく改善
- 議論
    - Disease mentionのうちのどの部分により着目すべきかは考慮できていない
- 次に読むべきもの
    - NCBI disease dataset について
    - Triplet network について

https://www.aclweb.org/anthology/W19-1912.pdf



###### 31. How to Fine-Tune BERT for Text Classification?

https://arxiv.org/abs/1905.05583


###### 30. Learning From Noisy Labels By Regularized Estimation Of Annotator Confusion. CVPR 2019.
- ラベルのノイズが大きい多クラス分類問題に対する改善手法を提案した論文. 
- 同一サンプルが複数人からアノテーションされている場合に有効.
- 損失関数を複雑にすることなく真のラベルと各アノテーターの特性を同時に推定する.
- 画像認識の論文だが様々な分野に適用可能と思われる
https://arxiv.org/abs/1902.03680
https://tech-blog.abeja.asia/entry/noisy-label-ml-survey


###### 29. Sara R et al. Leveraging Medical Literature for Section Prediction in Electronic Health Records. EMNLP 2019.
- どんな研究?
    - 診療記録の各文にアレルギー歴, A&P, 主訴, 検査結果, 家族歴, 身体所見, 薬剤歴, 既往歴, 社会歴, 処置, ROSを正しくラベル付けするタスク.
    - 次の2つを示した:
        - 目的とする診療記録データの少量にしかラベルが付いていないときは...
            - 診療記録ではないデータにラベル付けで訓練 + 目的のラベル付きデータでfine-tuningすると性能が向上した
        - 目的とする診療記録データに全くラベルが付いていないときは...
            - 診療記録ではないデータにラベル付けで訓練 + 他の診療記録ラベル付きデータでfine-tuningすると性能が向上した  

- 先行研究と比べて何がすごいか?
    - 手法の違い:
        - 先行研究: Sentence Classification は Sectionとそのheaderが同定された状態で行っている
        - 本研究: いきなり各文を分類させた  
        
- ラベルは次の11種類
    - アレルギー歴, A&P, 主訴, 検査結果, 家族歴, 身体所見, 薬剤歴, 既往歴, 社会歴, 処置, ROS
- データセットは次の3種類
    - MedLit: Medical Literature
        - textbook, guideline, medically relevant Wikipedia articles
        - train:val:test = 8:2:0 (EHRでの検証が目的なのでtestは行わない)
    - ClvC: Cleveland Clinic EHRs
        - 1施設54患者の178診療記録
        - アノテーターは医学生2名
            - 最初の106記録でκ=0.86であったため残りのデータは1名ずつにアノテーションさせた
        - train/val : test = 6:4
    - i2b2: i2b2 EHRs
        - 743 unique headers
        - i2b2 Risk Factors dataset
- 手法
    - train on ...
        - i) ClvC
        - ii) i2b2
        - iii) MedLit
        - iv) MedLit -> ClvC
        - v) MedLit -> i2b2
    - 
        - GRU
            - batch=32, dropout 0.2, 300d, Adam, 50epochs
        - BERT
            - batch=32, dropout 0.1, 768d, 128token
        - CNN, NB, SVMも試したがGRU, BERTより悪かった
    - 統計的有意差
        - McNemar's test
    - タスク
    -
    -
- 結果
    - section classification
        - test on ClvC:
            - train on ClvC only: BERT F1 0.89
            - train on MedLit + ClvC: BERT F1 0.90 (OK)
            - train on i2b2: BERT F1 0.83
            - train on MedLit + i2b2: BERT F1 0.78 (下がった)
        - test on i2b2:
            - train on i2b2: BERT F1 0.99
            - train on MedLit + i2b2: BERT F1 0.99 (不変)
            - train on ClvC only: BERT F1 0.84
            - train on MedLit + ClvC: BERT F1 0.92 (OK)
    - sentence classification
- Discussion
    - MedLit datasetの質
        - 自動でラベル付けしたのであまり性能は良くない
        - train on MedLit -> val on MedLitでBERT F1 72%にとどまった
        - 特にAssessment, Planが貧弱 (それはそう)
    
https://www.aclweb.org/anthology/D19-1492/

###### 28. Ofer Dekel et al. Optimal Distributed Online Prediction Using Mini-Batches. Journal of Machine Learning Research 2012; 13: 165-202.

https://arxiv.org/abs/1012.1367

###### 27. 





###### 26. A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature. (ACL 2018)

- EBM-NLPを紹介した論文.
- RCT論文からPICOを抽出するためのデータセット.
    - PubMed上のRCT論文5,000通から作成した.
        - 分野はcardiovascular, cancer and autism
- アノテーション手順は以下のとおり
    - 簡便性のためPICOのIとCは区別せずP,I,Oの3種類でtagging
    - ツールにはBRATを使用
    - stage 1 annotation:
        - P,I,Oのいずれかに該当する区間をすべてアノテート
    - stage 2 annotation:
        - P,I,Oそれぞれについて別々の人に以下をアノテート(認知的負荷とinstructionの負荷を減らすため)
            - 階層構造をもったタグ
            - repetition (情報の重複を検出するため)
            - MeSHタグの付与
    - クラウドソーシングした
        - non-expert worker
            - Amazon Mechanical Turk (AMT) を使用
        - expert worker
            - Up-work
- ベースライン
    - PIO tagging
        - biLSTM-CRF: F1 score 0.63-0.71
    - token level tagging
        - CRF: F1 score 0.21-0.55

https://www.aclweb.org/anthology/P18-1019


###### 25. SciBERT: A Pretrained Language Model for Scientific Text. (EMNLP/ICJNLP 2019)

- 科学技術分野に特化した事前学習済みBERT
- Semantic Scholarの約114万文献から事前学習した
    - 12%はcomputer science, 82%はさまざまなbiomedical domain
    - アブストだけではなく全文を使用
- 語彙はBERTと同じ語彙をWordPiece tokenizationしたものと, 科学技術コーパスにSentencePieceを適用して構成したものの2種類を用意
- 実験
    - EBM-NLP
    - ACL-ARC
    - SciCite
    - fine-tuningのハイパーパラメーターは最適なものを探索したが結局はBERTのものとほぼ類似のものに落ち着いた
    - 語彙を再構成し, かつEmbeddingも含めてfine-tuningした場合がもっとも好成績だった

https://www.aclweb.org/anthology/D19-1286/


###### 24. Axiomatic Attribution for Deep Networks. (ICML 2017)

変数の予測値への寄与度の計算法の1つであるIntegrated Gradientsの原著論文.
変数をベースラインの状態から目的の状態へと変化させたときの各変数の予測への寄与度を, 勾配値の積分によって計算したシャープレイ値として計算している.
ほとんどいたるところ全微分可能なモデルであれば種類を問わず適用可能. ライブラリも公開されている(https://github.com/ankurtaly/Integrated-Gradients). 説明はこちらに詳しい(https://qiita.com/ToshihiroNakae/items/c8604d19d48889be271c).

https://arxiv.org/pdf/1703.01365.pdf



###### 23. What does BERT look at? An Analysis of BERT's Attention. (ACL 2019)

- BERTの同一トークンまたは隣接トークンに対するattention
    - 4つのattention head (in layers 2,4,7,8) では50%以上が1つ前のトークンにあたる
    - 5つのattention head (in layers 1,2,2,3,6) では50%以上が1つ後ろのトークンにあたる
- BERT attentionの特定のトークンへの集中
    - Layer 1-3 では多くが [CLS] に当たっている
    - Layer 4-10 では多くが [SEP] に当たっている
        - しかしgradient-based measures of feature importanceを計算すると [SEP] に対するimportanceはLayer5以降ではかなり低い
        - よって [SEP] に当たっているAttentionは出力にあたって特に影響を及ぼさないと考えられる
    - Layer 11-12 では多くが .や,に当たっている
        - しかしgradient-based measures of feature importanceを計算すると .や, に対するimportanceはLayer11,12ではかなり低い
        - よって .や, に当たっているAttentionは出力にあたって特に影響を及ぼさないと考えられる
    - Attentionのエントロピーを計算した
        - エントロピーが高ければattentionは広く, 低ければ狭い
            - エントロピーはLayer1が最も高い
            - エントロピーの高いattention layerは1単語に当てるattentionは多くても10%であった
        - さらに[CLS]トークンからのみのattentionに対してエントロピーを計算
            - 最後のレイヤーでは[CLS]トークンからのattentionのエントロピーはかなり高かった 

https://www.aclweb.org/anthology/W19-4828/

###### 22. How multilingual is Multilingual BERT? (ACL 2019)

Multilingual BERTは事前学習に対訳コーパスを用いていないにも関わらず言語間でzero-shot transferができる(=言語Xのdatasetでfine tuning→言語Yのdatasetでtest). ただし語順の異なる言語間では性能が下がる

https://www.aclweb.org/anthology/P19-1493.pdf

###### 21. Automated Misspelling Detection and Correction in Persian Clinical Text. (JDI 2019)

ペルシャ語の超音波レポートの自動誤り訂正をn-gram言語モデルを用いた辞書ベース手法で実現した論文.
Accuracyは誤り検出で90％強, 誤り訂正で80％台

ncbi.nlm.nih.gov/pubmed/31823185



###### 20. MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. (Sci Data 2019)

MIMICから作った胸部単純X線画像(6.5万患者, 37.7万枚)+匿名化レポートのデータセット. 既存の胸部単純X線データセットもまとめて紹介している

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6908718/


###### 19. CliCR: A Dataset of Clinical Case Reports for Machine Reading Comprehension (NAACL 2018)

BMJ Case Reportsの症例報告約12,000件から作成した100,000件の空所補充問題データセット.
Stanford Attentive Reader, Gated-Attention ReaderではF1 30%前後

https://arxiv.org/pdf/1803.09720.pdf


###### 18. Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets. (ACL 2019)
医学領域の機械読解タスク(BLUE tasks)10種に対し, PubMed + MIMIC-III corpusで fine-tuningしたBERTとELMoによりSOTAを更新


###### 17. Preparing a collection of radiology examinations for distribution and retrieval. (JAMIA 2015)
胸部X線のレポートと画像の公開データセット.
読影レポートの匿名化は既存のシステム([Regenstrief Scrubber](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2923159/pdf/1471-2288-10-70.pdf))を用いてprecision 100%を達成.
またDICOMヘッダーと画像の匿名化はRSNA's Clinical Trials Processor & DICOM supplement 142 Clinical Trials De-identification methodologyにて行ったが0.5%ほどの症例で個人情報が残った.
さらにレポートへのタグ付与を次の2通りで行って症例検索に関する実験を行った.
- manual encoding: MeSHとRadLexのコードを人手で付与
- automatic encoding: MTIを用いた付与
実験用の症例検索クエリはImageCLEFのクエリを使用.
症例検索の性能は人手で評価した.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5009925/

###### 16. Deep Contextualized Biomedical Abbreviation Expansion. (ACL 2019)
医学用語の略語から元の語への正しい復元を試みた研究.
Med abstractから教師データを半自動的に作成し, BioELMo+BiLSTMによって分類器を作成. 一部のデータセットで人間の成績を上回った
https://www.aclweb.org/anthology/W19-5010/


###### 15. Natural language processing and machine learning algorithm to identify brain MRI reports with acute ischemic stroke (2019)
急性期梗塞疑い患者の頭部MRIレポートの陽性/陰性分類タスク. 決定木がF1 93%を達成. クラス不均衡補正を3通り試したが性能に変化なし
ncbi.nlm.nih.gov/pubmed/30818342


###### 14. HEAD-QA: A Healthcare Dataset for Complex Reasoning (ACL 2019)
スペインの医療従事者国家試験問題から作成した択一式QAデータセット.
Open-domainな質問応答システムでは正答率が低くなるようになっており, 正答にはドメイン知識が必要. スペイン語版と英語版がある
https://www.aclweb.org/anthology/P19-1092/


###### 13. Introducing Information Extraction to Radiology Information Systems to Improve the Efficiency on Reading Reports (2019)
肺CTの中国語レポートにBiGRU+CRFによる固有表現抽出を適用しF1 0.95を達成.
さらに検証実験では医師の情報抽出の所要時間を4割短縮させた
https://www.thieme-connect.com/products/ejournals/abstract/10.1055/s-0039-1694992


###### 12. Lessons from Natural Language Inference in the Clinical Domain. (EMNLP 2018)
臨床応用を想定した自然言語推論データセットMedNLIを作成した. 
前提文はMIMIC-IIIのカルテの既往歴から抜粋, 仮説文は医師にentail/contradict/neutralにあたる文を書いてもらい収集
https://www.aclweb.org/anthology/D18-1187.pdf


###### 11. Extracting relations between outcomes and significance levels in Randomized Controlled Trials (RCTs) publications.
RCT論文からアウトカムと有意差を抽出し, 正しい対応づけを試みた。アウトカム, 有意差, 対応づけのF値は79%(SciBERT), 98%(rule), 94%(BioBERT)
https://www.aclweb.org/anthology/W19-5038/


###### 10. Publicly Available Clinical BERT Embeddings. (NAACL 2019)
- ClinicalBERTはBioBERTをfine-tuningする形で事前学習したほうが医療言語処理タスクで好成績だった。
- また，事前学習コーパスにMIMIC-IIIのデータ全部ではなく退院サマリーだけを使っても性能はほぼ不変
https://www.aclweb.org/anthology/W19-1909/


###### 9. ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission. (2019)
MIMIC-IIIの入院患者診療記録を入力として事前学習したBERT。30日以内の再入院の有無をAUROC 0.674で予測できた。単語埋め込みとしての性能もFastTextやword2vecより高い
https://arxiv.org/abs/1904.05342


###### 8. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. (2019)
PubMed, PMCの論文コーパスで事前学習したBERT。
語彙はBERTから変更していないが，医療言語処理のNER, QA, REタスクでBERTより好成績 (BERT自体も健闘している)。
https://arxiv.org/abs/1901.08746


###### 7. Use of Machine Learning to Identify Follow-Up Recommendations in Radiology Reports. (JACR 2019)
読影レポートがfollow upを推奨しているかどうかRandom forest, ロジスティック回帰, SVM, RNNで分類. F1 scoreはSVMで0.85, RNNで0.71, 既存のiSCOUTシステムで0.71
https://www.sciencedirect.com/science/article/pii/S1546144018314042?via%3Dihub



###### 6. Extracting Follow-Up Recommendations and Associated Anatomy from Radiology Reports. (MEDINFO 2017)
読影レポートのうち画像でのfollow upを推奨しているものを固有表現抽出により分類. Accuracy 99%. Ground truthは放射線科医のアノテーション
http://ebooks.iospress.nl/publication/48323


###### 5. Automated Detection of Radiology Reports that Require Follow-up Imaging Using Natural Language Processing Feature Engineering and Machine Learning Classification. (J Digit Imag 2019)
腹部臓器の悪性腫瘍を示唆する読影レポートを決定木で分類. F1 0.458.
https://link.springer.com/article/10.1007%2Fs10278-019-00271-7


###### 4. Understanding Urgency in Radiology Reporting: Identifying Associations Between Clinical Findings in Radiology Reports and Their Prompt Communication to Referring Physicians.
緊急性の高い疾患を診断した読影レポートを, cTAKESによる固有表現抽出で検出.
https://www.ncbi.nlm.nih.gov/pubmed/31438224



###### 3. Show, Describe and Conclude: On Exploiting the Structure Information of Chest X-ray Reports (ACL 2019)
・強化学習や模倣学習により胸部単純X線レポート生成のSOTAを更新した論文
・Agentは行動選択(終了/正常所見入力/異常所見入力)と生成を反復. 報酬はBLEU-4
https://www.aclweb.org/anthology/P19-1657/


###### 2. Deep Contextualized Biomedical Abbreviation Expansion (ACL 2019)
・医学用語の略語からもとの語への正しい復元を人間よりも高い精度で実現した(Accuracy 98.4%)。
・教師データはPubMed abstractから自動収集。手法はBioELMo+BiLSTM
https://www.aclweb.org/anthology/W19-5010/



###### 1. Attention Is Not Explanation.
「Attention=判断根拠」を疑問視した論文。
文書分類, QA, NLIをBiLSTM+Attentionで解くと，
・AttentionはGBTのfeature importanceと一致しなかった。
・model outputをほぼ変えずにAttentionを大きく改変することが可能であった。
https://arxiv.org/abs/1902.10186
