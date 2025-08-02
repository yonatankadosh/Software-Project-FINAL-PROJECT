#!/bin/bash

set -uo pipefail

SCRIPT_DIR=`dirname $0`
PROJECT_DIR=${SCRIPT_DIR}/../src
TESTS_DIR=${SCRIPT_DIR}/tests/
KMEANS_TESTS_DIR=${SCRIPT_DIR}/kmeans_tests/

RED="\033[31m"
GREEN="\033[32m"
RESET="\033[0m"

failures=0
total_tests=0

function assertByFile() {
	command=$1
	expectedFilePath=$2
	name=$3

	FIXED_WIDTH=65

	((total_tests++))

	stderr_output=`mktemp /tmp/tmpfile.XXXXXX`
	stdout_output=`mktemp /tmp/tmpfile.XXXXXX`
	eval ${command} 2>${stderr_output} > ${stdout_output} || true
	stdout_with_line_numbers=`mktemp /tmp/tmpfile.XXXXXX`
	expected_with_line_numbers=`mktemp /tmp/tmpfile.XXXXXX`
	diff_output=`mktemp /tmp/tmpfile.XXXXXX`
	nl -v 0 ${stdout_output} > ${stdout_with_line_numbers}
	nl -v 0 ${expectedFilePath} > ${expected_with_line_numbers}
	printf "%-${FIXED_WIDTH}s %s" "${name}"
	diff ${expected_with_line_numbers} ${stdout_with_line_numbers} > ${diff_output} && echo -e "${GREEN}passed${RESET}" && return 0 || echo -e "${RED}FAILED${RESET}"
	((failures++))
	echo -e "\nFailed command: ${command}\n"
	echo "--- captured STDERR ---"
	cat ${stderr_output}
	echo "--- actual ---"
	echo original: ${stdout_output}
	echo with line numbers: ${stdout_with_line_numbers}
	echo "--- expected ---"
	echo original: ${expectedFilePath}
	echo with line numbers: ${expected_with_line_numbers}
	echo "--- diff ---"
	echo diff: ${diff_output}
	echo "________________"
	echo
	return 1
}

function testSymnmfSingleGoal() {
	goal=$1
	k=$2
	inputFilePath=${TESTS_DIR}/input_${3}.txt
	expectedFilePath=${TESTS_DIR}/output_${4}.txt
	mode=$5
	test_name=$6
	python_executable=${7:-symnmf.py}

	C_RUN_COMMAND="valgrind --quiet --leak-check=full ${PROJECT_DIR}/symnmf ${goal} ${inputFilePath}"
	PYTHON_RUN_COMMAND="python3 ${PROJECT_DIR}/${python_executable} ${goal} ${inputFilePath}"
	if [[ "$mode" == *"c"* ]]; then
		assertByFile "${C_RUN_COMMAND}" "${expectedFilePath}" 		"[C]        ${inputFilePath} - ${test_name}"
	fi
	if [[ "$mode" == *"python"* ]]; then
		assertByFile "${PYTHON_RUN_COMMAND}" "${expectedFilePath}" 	"[Python]   ${inputFilePath} - ${test_name}"
    fi
}

function testSymnmfGoalsWithK() {
	inputFileName=$1
	expectedFileName=$2
	k=$3
	name_ext=${4:-"k${k}"}
	
	# Test analysis.py with k parameter
	ANALYSIS_RUN_COMMAND="python3 ${PROJECT_DIR}/analysis.py ${k} ${TESTS_DIR}/input_${inputFileName}.txt"
	EXPECTED_FILE="${TESTS_DIR}/output_${expectedFileName}_analysis.txt"
	assertByFile "${ANALYSIS_RUN_COMMAND}" "${EXPECTED_FILE}" "[Analysis] input_${inputFileName}.txt - k=${k}"
}

function testSymnmfAllGoals() {
	inputFileName=$1
	expectedFileName=$2
	k=${3:-0}

	testSymnmfSingleGoal sym ${k} ${inputFileName} ${expectedFileName}_sym "python" sym
	testSymnmfSingleGoal ddg ${k} ${inputFileName} ${expectedFileName}_ddg "python" ddg
	testSymnmfSingleGoal norm ${k} ${inputFileName} ${expectedFileName}_norm "python" norm
	testSymnmfGoalsWithK $@ 2
}

function testKmeans() {
	k_and_maxIter=$1
	inputFilePath=${KMEANS_TESTS_DIR}/input_${2}.txt
	expectedFilePath=${KMEANS_TESTS_DIR}/output_${3}.txt

	KMEANS_RUN_COMMAND="python3 ${PROJECT_DIR}/kmeans.py ${k_and_maxIter} < ${inputFilePath}"
	assertByFile "${KMEANS_RUN_COMMAND}" ${expectedFilePath} "[Kmeans]   ${inputFilePath} - args: \"${k_and_maxIter}\""
}

function testAnalysis() {
	k=$1
	inputFilePath=${TESTS_DIR}/input_${2}.txt
	expectedFilePath=${TESTS_DIR}/output_${3}_analysis.txt

	ANALYSIS_RUN_COMMAND="python3 ${PROJECT_DIR}/analysis.py ${k} ${inputFilePath}"
	assertByFile "${ANALYSIS_RUN_COMMAND}" ${expectedFilePath} "[Analysis] ${inputFilePath} - k=${k}"
}

function symnmfSuite() {
	echo ">>> SymNMF <<<"

	# Test basic functionality
	testSymnmfAllGoals 1 1
	testSymnmfAllGoals 2 2
	testSymnmfAllGoals 3 3
	testSymnmfAllGoals 6 6
	testSymnmfAllGoals 7 7
	testSymnmfAllGoals 8 8

	echo
}

function kmeansSuite() {
	echo ">>> Kmeans <<<"

	testKmeans "3 600" 1 1
	testKmeans "7" 2 2
	testKmeans "15 300" 3 3

	echo
}

function analysisSuite() {
	echo ">>> Analysis <<<"

	testAnalysis 2 1 1
	testAnalysis 7 2 2
	testAnalysis 15 3 3
	testAnalysis 2 6 6
	testAnalysis 7 7 7
	testAnalysis 8 8 8

	echo
}

function cleanTmp() {
	if [ "$(ls -1 /tmp/tmpfile.?????? | wc -l)" -gt 600 ]; then
		echo -e "\n! Removing old temporary files !\n"
		ls -tr /tmp/tmpfile.?????? | head -n 300 | while read file; do rm "$file"; done || true
	fi
}

cleanTmp

echo "Testing original project files in: ${PROJECT_DIR}"
echo

kmeansSuite
symnmfSuite
analysisSuite

if [ $failures -eq 0 ]; then
	echo -e "\n${GREEN}=== ${total_tests}/${total_tests} tests passed successfully ===${RESET}\n"
else
	echo -e "\n${RED}=== ${failures}/${total_tests} tests failed ===${RESET}\n"
	exit 1
fi 