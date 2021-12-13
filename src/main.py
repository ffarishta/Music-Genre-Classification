import nn_baseline
import nn
import knn
import logistic_regression


def main():
    """
    Main function will run a command line interface to implment the chosen classifiers  
    """
    while(True):
        #choose 1 of 4 options 

        print("Classifiers for music genre classification")
        print("Enter 1 for KNN")
        print("Enter 2 for LG")
        print("Enter 3 for NN")
        print("Enter 4 to quit")
        val = int(input("Select The model that you would like to run: "))
        if val == 4:
            break
        reg = int(input("Enter 1 for baseline model or enter 2 for final model: "))
        
        # run each model given the input 
        if val == 1:
            if reg == 1:
                knn.KNN_baseline()
            if reg == 2:
                print("hi")
                knn.KNN_final()

        elif val == 3:
            if reg == 1:
                nn_baseline.baseline_nn_model()
            if reg == 2:
                nn.nn_model()
        
        elif val == 2:
            if reg == 1:
                logistic_regression.LR_baseline()
            if reg == 2:
                logistic_regression.LR_final()
        
        else:
            print("INVALID COMMAND! PLEASE TRY")
        
        print("\n")
            

if __name__ == "__main__":
    main()