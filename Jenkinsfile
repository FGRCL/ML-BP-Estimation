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
                        [string(variable:'wandb-api-key', credentialsId:'wandb-api-key')]
                    ){
                         def secrets = [
                            wandbApiKey:'$wandb-api-key'
                        ]
                        environmentVariables = renderTemplate(readFile("environments/${DEPLOYMENT_ENVIRONMENT}/template.env"), secrets)
                        writeFile("environments/${DEPLOYMENT_ENVIRONMENT}/variables.env", environmentVariables.toString())
                        archiveArtifacts("environments/${DEPLOYMENT_ENVIRONMENT}/variables.env")
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
                    sh """
                        ssh fgrcl@cedar.computecanada.ca mkdir ${SCRIPT_PATH}
                        scp ${SCRIPT_NAME} fgrcl@cedar.computecanada.ca:${SCRIPT_PATH}
                        scp environments/${DEPLOYMENT_ENVIRONMENT}/variables.env ${SCRIPT_PATH}
                        ssh fgrcl@cedar.computecanada.ca <<- EOF
                            cd ${SCRIPT_PATH}
                            chmod +x ${SCRIPT_NAME}
                            sbatch --export=IMAGE_TAG=${IMAGE_TAG} ${SCRIPT_NAME}
                        EOF
                    """.stripIndent()
                }
            }
        }
    }
}