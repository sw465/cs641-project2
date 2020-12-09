
# Cloud Computing Project 2

# Model training of 4 EC2 instances using AWS EMR

## Login to AWS Console
## Search for EMR
## Click "Create Cluster" button
## Click "Go to advanced options"
## Select "emr-6.2.0" release
## Check off "Spark 3.0.1" and "Hadoop 3.2.1" and click next
## Use 3 instances for Core - 2
## Type a name for the EMR cluster (e.g. modelTraining) and click next
## Choose an EC2 key pair
## Click "Create Cluster" button


## Open a terminal and connect to EMR EC2 master instance (Can be found under EMR cluster, summary, Master public DNS)

```ssh -i <Path to EC2 pem key>/<Your key>.pem  hadoop@<Master public DNS>```

## From a seperate terminal, upload trainModel.py with:

```scp -i <Path to EC2 pem key>/<Your key>.pem <local path to file>/trainModel.py  hadoop@<Master public DNS>:.```

## Create an S3 bucket from the AWS console to store the TrainingDataset.csv and the ML model to be created

## Now run trainModel.py from the ec2 EMR terminal, passing the TrainingDataset and the model storage location for arguments 1 and 2, respectively:

```spark-submit trainModel.py s3://training-dataset-643/TrainingDataset.csv s3://wine-model-643/rfmodel.model```

## Terminate instances from AWS console when finished 





# Getting prediction from saved model on EC2


## Create an EMR cluster again as before, but chose 0 instances for Core - 2, leaving only the single master instance

## Connect to EMR EC2 instance as before:

```ssh -i <Path to EC2 pem key>/<Your key>.pem  hadoop@<Master public DNS>```

## From a seperate terminal, upload predictModel.py with:

```scp -i <Path to EC2 pem key>/<Your key>.pem <local path to file>/predictModel.py  hadoop@<Master public DNS>:.```

## Upload the ValidationDataset.csv into the same S3 bucket created before.

## Now run predictModel.py from the ec2 EMR terminal, passing the ValidationDataset (from S3 bucket) and the model storage location (S3 bucket) for arguments 1 and 2, respectively:

```spark-submit predictModel.py s3://training-dataset-643/ValidationDataset.csv s3://wine-model-643/rfmodel.model```


## Look for f1 score printed to console

## Terminate instances from AWS console when finished 


# Runner docker on EC2

## Login to EC2 instance with:

```sh -i <Path to EC2 pem key>/<Your key>.pem  ec2-user@<EC2 DNS IP>```

## From a seperate terminal, upload your "TestDataset.csv" with:

```scp -i <Path to EC2 pem key>/<Your key>.pem <local path to file>/TestDataset.csv  ec2-user@<EC2 DNS IP>:.```


## On the terminal connected to EC2 instane, install docker:


```
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -aG docker ec2-user
```


## Confirm docker by running:

```docker ps```

## Pull docker image from repo:

```docker pull seanc23/cs641-project2```

## Run docker and pass in TestDataset.csv via docker volume with:

docker run -v <ec2 path to file>/TestDataset.csv:/data/TestDataset.csv seanc23/cs641-project2:latest TestDataset.csv

## Look for f1 score printed to console
