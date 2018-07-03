using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork myFirstNeuralNetWork = new NeuralNetwork(2, 12, 22);
            float[] inputArr = { 1, 0 };
            float[] outputArr = myFirstNeuralNetWork.FeedForward(inputArr);

            for (int i = 0; i < outputArr.Length; i++)
            {
                Console.WriteLine(outputArr[i]);
            }
        }
    }
}
