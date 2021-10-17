import unittest
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'adjoint_state_method'))
sys.path.append(os.path.join(os.getcwd(), 'dadi_torch'))
sys.path.append(os.path.join(os.getcwd(), 'models'))
print("sys.path:", sys.path)
import test_optimize1D

# calcTestSuite = unittest.TestSuite()
# # calcTestSuite.addTest(unittest.makeSuite(test_optimize.TestOptimizeParamsBFGS))
# calcTestSuite.addTest(unittest.makeSuite(test_optimize.TestOptimizeParamsGradDescent))
# print("count of tests: " + str(calcTestSuite.countTestCases()) + "\n")
# runner = unittest.TextTestRunner(verbosity=2)
# runner.run(calcTestSuite)

testLoad = unittest.TestLoader()
suites = testLoad.loadTestsFromName('test_optimize1D.TestOptimizeParamsGradDescent.test_optimize_two_epoch')
# suites = testLoad.loadTestsFromName('tests.test_optimize2D.TestOptimizeParamsGradDescent.test_optimize_bottlegrowth')
# suites = testLoad.loadTestsFromName(
# 'tests.test_optimize.TestOptimizeParamsGradDescent.test_optimize_grad_two_epoch_ASM')
# suites = testLoad.loadTestsFromName(
# 'tests.test_optimize.TestOptimizeParamsGradDescent.test_optimize_three_epoch_ASM')
# suites = testLoad.loadTestsFromName('tests.test_optimize.TestOptimizeParamsBFGS.test_optimize_two_epoch_ASM')
# suites = testLoad.loadTestsFromName('tests.test_optimize.TestOptimizeParamsBFGS.test_optimize_simple_1D')
# suites = testLoad.loadTestsFromName('tests.test_optimize.TestOptimizeParamsBFGS.test_optimize_three_epoch_ASM')
runner = unittest.TextTestRunner(verbosity=1)
runner.run(suites)

