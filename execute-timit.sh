# Estrazione delle features tramite il file di configurazione a piacere
./extract-corpus.sh -c PATH/TIMIT/TEST/ -f <OPENSMILE_CONF_FILE>
./extract-corpus.sh -c PATH/TIMIT/TRAIN/ -f <OPENSMILE_CONF_FILE>
# NOTA: Dopo averli eseguiti saranno creati dei dati temporanei dentro "rawdata/timit/"

# Prepara il dataset da dare alla rete neurale tramite 6 core, cioè labeling di ogni frame per ogni espressione, processo CPU
# per spiegazione python3.5 prepare-datasets.py -h
# comando pronto
python3.5 prepare-datasets.py -d rawdata/timit/TEST/ -o test.csv -c 6
python3.5 prepare-datasets.py -d rawdata/timit/TRAIN/ -o train.csv -c 6

# Lanciare l'apprendimento
python3.5 train-model.py -n "nome-sessione" -m "td-blstm" -t datasets/<NOME_TRAINSET.csv> -v 0.2 -s datasets/<NOME_TESTSET.csv> -c 6


# Alla fine di ogni validazione verranno fuori le metriche implementate per il progetto second quanto indicato nel libro (fscore e altri)
# Tutte le metriche che iniziano con "nlp_*" sono state implementate per questo progetto
# A ogni sessione va assegnato un nome, e nella cartella con tale nome ci saranno tutte le statistiche visualizzabili anche con TENSOR BOARD in tempo reale

# segue l'help per i comandi
#   -h, --help            show this help message and exit
#   -n SESSION_NAME, --session_name SESSION_NAME
#                         Name of folder where models will be saved
#   -m {td-blstm}, --model_type {td-blstm}
#                         Specify model type, 'td-blstm' for time distributed
#                         blstm, 'mtd-blstm' for time distributed blstm with
#                         masked input
#   -t TRAINSET, --trainset TRAINSET
#                         path to trainset .csv file
#   -v {0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35}, --static_validation {0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35}
#                         use static validation set shrinked from trainset
#   -s TESTSET, --testset TESTSET
#                         path to testset .csv file
#   -c {1,2,3,4,5,6,7}, --cores {1,2,3,4,5,6,7}
#                         Number of physical core to use

