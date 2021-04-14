import unittest
from tests import test_optimize

# calcTestSuite = unittest.TestSuite()
# # calcTestSuite.addTest(unittest.makeSuite(test_optimize.TestOptimizeParamsBFGS))
# calcTestSuite.addTest(unittest.makeSuite(test_optimize.TestOptimizeParamsGradDescent))
# print("count of tests: " + str(calcTestSuite.countTestCases()) + "\n")
# runner = unittest.TextTestRunner(verbosity=2)
# runner.run(calcTestSuite)

testLoad = unittest.TestLoader()
suites = testLoad.loadTestsFromName('tests.test_optimize.TestOptimizeParamsGradDescent.test_optimize_simple1D_model')
# suites = testLoad.loadTestsFromName(
# 'tests.test_optimize.TestOptimizeParamsGradDescent.test_optimize_grad_two_epoch_ASM')
# suites = testLoad.loadTestsFromName(
# 'tests.test_optimize.TestOptimizeParamsGradDescent.test_optimize_three_epoch_ASM')
# suites = testLoad.loadTestsFromName('tests.test_optimize.TestOptimizeParamsBFGS.test_optimize_two_epoch_ASM')
# suites = testLoad.loadTestsFromName('tests.test_optimize.TestOptimizeParamsBFGS.test_optimize_simple_1D')
# suites = testLoad.loadTestsFromName('tests.test_optimize.TestOptimizeParamsBFGS.test_optimize_three_epoch_ASM')
runner = unittest.TextTestRunner(verbosity=1)
runner.run(suites)

