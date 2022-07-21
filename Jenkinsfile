pipeline{
    agent any
    environment {
        IMAGE_TAG = "train-test"
        DOCKERHUB_PROJECT = "fgrcl/ml-bp-estimation"
        GIT_URL = "github.com:FGRCL/ML-BP-Estimation.git"
        GIT_BRANCH = "docker"
        SCRIPT_NAME = "test.sh"
    }
    stages {
        stage('Clone repo') {
            steps {
                git credentialsId: 'ssh-key', url: "git@${GIT_URL}", branch: "${GIT_BRANCH}"
            }
        }
        stage('Build image') {
            steps {
                sh "docker build -t ${DOCKERHUB_PROJECT}:${IMAGE_TAG} ."
            }
        }
        stage('Publish image') {
            environment {
                DOCKERHUB_CREDENTIALS = credentials('docker-credentials')
            }
            steps {
                sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
                sh "docker push ${DOCKERHUB_PROJECT}:${IMAGE_TAG}"
            }
        }
        stage('Train') {
            environment {
                SCRIPT_PATH = "~/projects/def-bentahar/fgrcl/jenkins"
            }
            steps {
                sshagent(credentials: ['ssh-key-cc']){
                    sh """
                        scp ${SCRIPT_NAME} fgrcl@cedar.computecanada.ca:${SCRIPT_PATH}
                        ssh fgrcl@cedar.computecanada.ca "cd ${SCRIPT_PATH} && chmod +x ${SCRIPT_NAME} && srun ${SCRIPT_NAME}"
                    """
                }
            }
        }
    }
}