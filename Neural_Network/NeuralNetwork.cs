using MathNet.Numerics.LinearAlgebra;
using System.Numerics;

namespace Neural_Network
{
    class NeuralNetwork
    {
        int m_InputNodes;
        int m_HiddenNodes;
        int m_OutputNodes;

        Vector<float> m_InputVector;
        Vector<float> m_InputToHiddenBias;

        Vector<float> m_OutputVector;
        Vector<float> m_HiddenToOutputBias;

        Matrix<float> m_InputToHidden_WeightsMatrix;
        Matrix<float> m_HiddenToOutput_WeightsMatrix;

        Vector<float> m_HiddenVector;

        public NeuralNetwork(int i_NumberOfInputs, int i_NumberOfHiddenNodes, int i_NumberOfOutputNodes)
        {
            m_InputNodes = i_NumberOfInputs;
            m_HiddenNodes = i_NumberOfHiddenNodes;
            m_OutputNodes = i_NumberOfOutputNodes;

            // Now init all the matrixes size.
            m_InputVector = Vector<float>.Build.Dense(i_NumberOfInputs);
            m_OutputVector = Vector<float>.Build.Dense(i_NumberOfOutputNodes);
            m_HiddenVector = Vector<float>.Build.Dense(i_NumberOfHiddenNodes);

            m_InputToHiddenBias = Vector<float>.Build.Random(i_NumberOfHiddenNodes);
            m_HiddenToOutputBias = Vector<float>.Build.Random(i_NumberOfOutputNodes);

            m_InputToHidden_WeightsMatrix = Matrix<float>.Build.Random(i_NumberOfHiddenNodes, i_NumberOfInputs);
            m_HiddenToOutput_WeightsMatrix = Matrix<float>.Build.Random(i_NumberOfOutputNodes, i_NumberOfHiddenNodes);

        }

        public float[] FeedForward(float[] input)
        {
            //Inserting the input array to the input vector.
            m_InputVector.SetValues(input);

            //Generate hidden vector values (first multypliy, then add bias. then use sigmoid to normiliztion)
            m_HiddenVector = (m_InputToHidden_WeightsMatrix.Multiply(m_InputVector));
            m_HiddenVector.Add(m_InputToHiddenBias);
            m_HiddenVector.Map(Sigmoid);

            //Same thing as above for Output vector.
            m_OutputVector = (m_HiddenToOutput_WeightsMatrix.Multiply(m_HiddenVector));
            m_OutputVector.Add(m_HiddenToOutputBias);
            m_OutputVector.Map(Sigmoid);

            //Returing the output vector as array.
            return m_OutputVector.ToArray();
        }

        private double Sigmoid(float x)
        {
            return 1 / (1 + System.Math.Exp(-x));
        }
    }
}
