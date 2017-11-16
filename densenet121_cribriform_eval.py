    # Save Format:
    # 2D feature
    # <image_label> <image_feature[0]> <image_feature[1]>
    # for both training and testing data
    print('Saving features from DenseNet-121 for Fold %d:' % fold)

    # Make predictions on train data
    predictions_train = model.predict(X_train, batch_size=batch_size, verbose=0)
    with open('features/fold_%d_train_results.txt' % fold, 'w') as outfile:
        for (y_, y) in zip(Y_train, predictions_valid):
            if y_[0] == 1:
                outfile.write('1 ')
            else:
                outfile.write('0 ')
            for yi in y:
                outfile.write(str(yi) + ' ')
            outfile.write('\n')

    # Make predictions on test data
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    with open('features/fold_%d_test_results.txt' % fold, 'w') as outfile:
        for (y_, y) in zip(Y_valid, predictions_valid):
            if y_[0] == 1:
                outfile.write('1 ')
            else:
                outfile.write('0 ')
            for yi in y:
                outfile.write(str(yi) + ' ')
            outfile.write('\n')

    print('Done saving features')
