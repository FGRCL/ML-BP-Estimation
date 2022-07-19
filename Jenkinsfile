pipeline{
    agent none
    environment {
        IMAGE_TAG = "train-test"
        DOCKERHUB_PROJECT = "fgrcl/ml-bp-estimation"
        GIT_URL = "github.com:FGRCL/ML-BP-Estimation.git"
        GIT_BRANCH = "main"
    }
    stages {
        stage('Clone repo') {
            agent any
            steps {
                git credentialsId: 'ssh-key', url: "git@${GIT_URL}", branch: "${GIT_BRANCH}"
            }
        }
        stage('Container'){
            agent any
            environment {
                DOCKERHUB_CREDENTIALS = credentials('docker-credentials')
            }
            steps {
                //sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
                sh 'ls'
                sh 'docker build -t ${DOCKERHUB_PROJECT}:${IMAGE_TAG} .'
                sh 'docker push ${DOCKERHUB_PROJECT}:${IMAGE_TAG}'
            }
        }
        stage('Train'){
            agent{
                label 'CCCedar'
            }
            steps{
                git branch: '${env.BRANCH_NAME}',
                    credentialsId: 'Jenkins',
                    url: 'ssh://git@github.com:FGRCL/ML-BP-Estimation.git'
                sh 'sbatch ./train.sh'
            }
        }
    }
}