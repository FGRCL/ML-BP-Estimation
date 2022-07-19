pipeline{
    agent none
    environment {
        IMAGE_TAG = "train-test"
        DOCKERHUB_PROJECT = "fgrcl/ml-bp-estimation"
        GIT_URL = "github.com:FGRCL/ML-BP-Estimation.git"
        GIT_BRANCH = "docker"
    }
    stages {
        stage('Clone repo') {
            agent any
            steps {
                git credentialsId: 'ssh-key', url: "git@${GIT_URL}", branch: "${GIT_BRANCH}"
            }
        }
        stage('Build image') {
            agent any
            steps {
                sh 'docker build -t ${DOCKERHUB_PROJECT}:${IMAGE_TAG} .'
            }
        }
        stage('Publish image') {
            agent any
            environment {
                DOCKERHUB_CREDENTIALS = credentials('docker-credentials')
            }
            steps {
                sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
                sh "docker push ${DOCKERHUB_PROJECT}:${IMAGE_TAG}"
            }
        }
        stage('Train') {
            agent{
                label 'CCCedar'
            }
            steps{
                git credentialsId: 'ssh-key', url: "git@${GIT_URL}", branch: "${GIT_BRANCH}"
                sh 'sbatch ./train.sh'
            }
        }
    }
}