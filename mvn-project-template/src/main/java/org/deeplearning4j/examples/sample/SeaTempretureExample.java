/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.examples.sample;

import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.GradientNormalization;
// import org.deeplearning4j.nn.conf.layers;
//import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.Updater;

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;

import java.io.File;
import java.net.URL;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;

/**
 * Created by adielas on 30/01/21.
 */
public class SeaTempretureExample {
    private static final Logger log = LoggerFactory.getLogger(SeaTempretureExample.class);

    public static void main(String[] args) throws Exception {


        /*
        Download data from url
         */

        boolean shouldDownload = false;

        String DATA_URL = "https://dl4jdata.blob.core.windows.net/training/seatemp/sea_temp.tar.gz";
        String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_seas/");

        if(shouldDownload) {

            File directory = new File(DATA_PATH);
            directory.mkdir();

            String archizePath = DATA_PATH + "sea_temp.tar.gz";
            File archiveFile = new File(archizePath);
            String extractedPath = DATA_PATH + "sea_temp";
            File extractedFile = new File(extractedPath);

            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);

            int fileCount = 0;
            int dirCount = 0;
            int BUFFER_SIZE = 4096;

            TarArchiveInputStream tais = new TarArchiveInputStream(new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(archizePath))));

            ArchiveEntry entry = tais.getNextEntry();

            while (entry != null) {
                if (entry.isDirectory()) {
                    new File(DATA_PATH + entry.getName()).mkdirs();
                    dirCount = dirCount + 1;
                    fileCount = 0;
                } else {
                    byte[] data = new byte[4 * BUFFER_SIZE];
                    FileOutputStream fos = new FileOutputStream(DATA_PATH + entry.getName());
                    BufferedOutputStream dest = new BufferedOutputStream(fos, BUFFER_SIZE);
                    int count;

                    while ((count = tais.read(data)) > 0) {
                        dest.write(data, 0, count);
                    }

                    dest.flush();
                    dest.close();
                    fileCount = fileCount + 1;
                }
                if (fileCount % 1000 == 0) {
                    log.info(".");
                }

                entry = tais.getNextEntry();
            }

        }
        /*
            Read data using CSVSequenceRecordReader
         */

        String path = FilenameUtils.concat(DATA_PATH, "sea_temp/"); // set parent directory

        String featureBaseDir = FilenameUtils.concat(path, "features"); // set feature directory
        String targetsBaseDir = FilenameUtils.concat(path, "targets"); // set label directory

        int numSkipLines = 1;
        boolean regression = true;
        int batchSize = 32;

        CSVSequenceRecordReader trainFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
        trainFeatures.initialize( new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 1, 1042));
        CSVSequenceRecordReader trainTargets = new CSVSequenceRecordReader(numSkipLines, ",");
        trainTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", 1, 1042));

        SequenceRecordReaderDataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainTargets, batchSize,
            10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);


        CSVSequenceRecordReader testFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
        testFeatures.initialize( new NumberedFileInputSplit(featureBaseDir + "/%d.csv", 1043, 1736));
        CSVSequenceRecordReader testTargets = new CSVSequenceRecordReader(numSkipLines, ",");
        testTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", 1043, 1736));

        SequenceRecordReaderDataSetIterator test = new SequenceRecordReaderDataSetIterator(testFeatures, testTargets, batchSize,
            10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);



        /*
            Init neuarl network
         */

        int V_HEIGHT = 13;
        int V_WIDTH = 4;
        int kernelSize = 2;
        int numChannels = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(new AdaGrad(0.005))
            .list()
            .layer(0, new ConvolutionLayer.Builder(kernelSize, kernelSize)
                .nIn(1) //1 channel
                .nOut(7)
                .stride(2, 2)
                .activation(Activation.RELU)
                .build())
            .layer(1, new LSTM.Builder()
                .activation(Activation.SOFTSIGN)
                .nIn(84)
                .nOut(200)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10)
                .build())
            .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(200)
                .nOut(52)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10)
                .build())
            .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
            .inputPreProcessor(1, new CnnToRnnPreProcessor(6, 2, 7 ))
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // Train model on training set
        net.fit(train , 25);

        net.save(new File("/Users/ashrov/Documents/Dropbox/Adiel Guy/DNNProject/sea_temp_model.net"));

        RegressionEvaluation eval = net.evaluateRegression(test);

        test.reset();
        System.out.println();
        System.out.println(eval.stats());

        log.info("Finished evaluation....");

        log.info("****************Example finished********************");
    }
}
