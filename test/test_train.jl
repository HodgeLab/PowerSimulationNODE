p = NODETrainParams(base_path = TEST_FILES_DIR, verify_psid_node_off = false)
status = train(p)
@test status 
