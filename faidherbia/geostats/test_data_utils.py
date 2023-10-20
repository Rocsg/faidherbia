import faidherbia
import faidherbia.geostats
import faidherbia.geostats.data_utils as data_utils

"""
-------- To run in shell, copy this code --------
python
import faidherbia.geostats.test_data_utils
faidherbia.geostats.test_data_utils.run_test()
-------------------------------------------------
""" 

def run_test():
    print("Testing data_utils.is_mansour...")
    print("is_mansour()="+str(data_utils.is_mansour()))
    print("Test passed !")


    print("Testing data_utils.get_main_directory...")
    print("get_main_directory()="+str(data_utils.get_main_directory()))
    print("Test passed !")

    print("Testing data_utils.get_aerial_imaging_dates...")
    print("get_aerial_imaging_dates()="+str(data_utils.get_aerial_imaging_dates(it_is_just_a_test=True)))
    print("Test passed !")


    print("Testing data_utils.get_parcel_list...")
    print("get_parcel_list()="+str(data_utils.get_parcel_list(it_is_just_a_test=True)))
    print("Test passed !")


    print("Testing data_utils.create_result_dirs...")
    datadir=data_utils.get_main_directory()
    years,dates=data_utils.get_aerial_imaging_dates(it_is_just_a_test=True)
    parcels=data_utils.get_parcel_list(it_is_just_a_test=True)
    data_utils.create_result_dirs(datadir,years,dates,parcels)
    print("")
    print("######################################################")
    print("#                   Test passed !                    #")
    print("######################################################")
    print("")

run_test()