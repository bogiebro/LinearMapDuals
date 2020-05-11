using ReTester, Revise
using Tests
entr(run_tests(Tests.test), ["src", "Tests/src"]; pause=0.5)
