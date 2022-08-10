import groovy.text.StreamingTemplateEngine

if (currentBuild.getBuildCauses().toString().contains('BranchIndexingCause')) {
  print "INFO: Build skipped due to trigger being Branch Indexing"
  currentBuild.result = 'ABORTED' // optional, gives a better hint to the user that it's been skipped, rather than the default which shows it's successful
  return
}

def renderTemplate(input, variables) {
  def engine = new StreamingTemplateEngine()
  return engine.createTemplate(input).make(variables).toString()
}

pipeline{
    agent any
    environment {
        IMAGE_TAG = "${env.BRANCH_NAME}-${currentBuild.id}"
        DOCKERHUB_PROJECT = "fgrcl/ml-bp-estimation"
        GIT_URL = "github.com:FGRCL/ML-BP-Estimation.git"
        SCRIPT_PATH = "/home/fgrcl/projects/def-bentahar/fgrcl/jenkins/${IMAGE_TAG}"
        SCRIPT_NAME = "train.sh"
        DEPLOYMENT_ENVIRONMENT = "cedar.computecanada.ca"
    }
    stages {
        stage('Clone repo') {
            steps {
                git credentialsId: 'ssh-key', url: "git@${GIT_URL}", branch: "${env.BRANCH_NAME}"
            }
        }
        stage('Replace secrets') {
            steps {
                script {
                    withCredentials(
                        [string(variable:'WANDB_API_KEY', credentialsId:'wandb-api-key')]
                    ){
                        def secrets = [
                            WANDB_API_KEY: env.WANDB_API_KEY
                        ]
                        def templateFile = readFile("environments/${DEPLOYMENT_ENVIRONMENT}/template.env")
                        def environmentVariables = renderTemplate(templateFile.toString(), secrets)
                        writeFile(file: "environments/${DEPLOYMENT_ENVIRONMENT}/variables.env", text: environmentVariables.toString())
                    }
                }
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
                    sh '''
                        ssh fgrcl@cedar.computecanada.ca mkdir ${SCRIPT_PATH}
                        scp ${SCRIPT_NAME} fgrcl@cedar.computecanada.ca:${SCRIPT_PATH}
                        scp environments/${DEPLOYMENT_ENVIRONMENT}/variables.env fgrcl@cedar.computecanada.ca:${SCRIPT_PATH}
                        ssh fgrcl@cedar.computecanada.ca <<- EOF
                            cd ${SCRIPT_PATH}
                            chmod +x ${SCRIPT_NAME}
                            ./${SCRIPT_NAME} ${IMAGE_TAG}
                        EOF
                    '''.stripIndent()
                }
            }
        }
    }
}