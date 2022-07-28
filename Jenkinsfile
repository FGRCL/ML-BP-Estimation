pipeline{
    agent any
    environment {
        IMAGE_TAG = "${env.BRANCH_NAME}-${currentBuild.id}"
        DOCKERHUB_PROJECT = "fgrcl/ml-bp-estimation"
        GIT_URL = "github.com:FGRCL/ML-BP-Estimation.git"
        GIT_BRANCH = "docker"
        SCRIPT_PATH = "/home/fgrcl/projects/def-bentahar/fgrcl/jenkins/${IMAGE_TAG}"
        SCRIPT_NAME = "train.sh"
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
            steps {
                sshagent(credentials: ['ssh-key-cc']){
                    sh """
                        ssh fgrcl@cedar.computecanada.ca mkdir ${SCRIPT_PATH}
                        scp ${SCRIPT_NAME} fgrcl@cedar.computecanada.ca:${SCRIPT_PATH}
                        ssh fgrcl@cedar.computecanada.ca /bin/bash <<< EOF
                            cd ${SCRIPT_PATH}
                            chmod +x ${SCRIPT_NAME}
                            sbatch ${SCRIPT_NAME} --export=IMAGE_TAG=${IMAGE_TAG}
                        EOF
                    """
                }
            }
        }
    }
}