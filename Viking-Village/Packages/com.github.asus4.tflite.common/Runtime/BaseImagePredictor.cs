﻿using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;

namespace TensorFlowLite
{
    public abstract class BaseImagePredictor<T> : System.IDisposable
        where T : struct
    {
        protected readonly Interpreter interpreter;
        protected readonly int width;
        protected readonly int height;
        protected readonly int channels;
        protected readonly T[,,] input0;
        protected readonly TextureToTensor tex2tensor;
        protected readonly TextureResizer resizer;
        protected TextureResizer.ResizeOptions resizeOptions;

        public Texture inputTex
        {
            get
            {
                return (tex2tensor.texture != null)
                    ? tex2tensor.texture as Texture
                    : resizer.texture as Texture;
            }
        }
        public Material transformMat => resizer.material;

        public TextureResizer.ResizeOptions ResizeOptions
        {
            get => resizeOptions;
            set => resizeOptions = value;
        }

        public BaseImagePredictor(string modelPath, bool useGPU = true)
        {
            var options = new InterpreterOptions();
            if (useGPU)
            {
                options.AddGpuDelegate();
            }
            else
            {
                options.threads = SystemInfo.processorCount;
            }

            try
            {
                interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            }
            catch (System.Exception e)
            {
                interpreter?.Dispose();
                throw e;
            }

            interpreter.LogIOInfo();
            // Initialize inputs
            {
                var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
                height = inputShape0[1];
                width = inputShape0[2];
                channels = inputShape0[3];
                input0 = new T[height, width, channels];

                int inputCount = interpreter.GetInputTensorCount();
                for (int i = 0; i < inputCount; i++)
                {
                    int[] shape = interpreter.GetInputTensorInfo(i).shape;
                    interpreter.ResizeInputTensor(i, shape);
                }
                interpreter.AllocateTensors();
            }

            tex2tensor = new TextureToTensor();
            resizer = new TextureResizer();
            resizeOptions = new TextureResizer.ResizeOptions()
            {
                aspectMode = AspectMode.Fill,
                rotationDegree = 0,
                mirrorHorizontal = false,
                mirrorVertical = false,
                width = width,
                height = height,
            };
        }

        public virtual void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
            resizer?.Dispose();
        }

        public abstract void Invoke(Texture inputTex);

        protected void ToTensor(Texture inputTex, float[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(RenderTexture inputTex, float[,,] inputs, bool resize)
        {
            RenderTexture tex = resize ? resizer.Resize(inputTex, resizeOptions) : inputTex;
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(Texture inputTex, float[,,] inputs, float offset, float scale)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs, offset, scale);
        }

        protected void ToTensor(Texture inputTex, sbyte[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(Texture inputTex, int[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        protected async UniTask<bool> ToTensorAsync(Texture inputTex, float[,,] inputs, CancellationToken cancellationToken)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            await tex2tensor.ToTensorAsync(tex, inputs, cancellationToken);
            return true;
        }

        protected async UniTask<bool> ToTensorAsync(RenderTexture inputTex, float[,,] inputs, bool resize, CancellationToken cancellationToken)
        {
            RenderTexture tex = resize ? resizer.Resize(inputTex, resizeOptions) : inputTex;
            await tex2tensor.ToTensorAsync(tex, inputs, cancellationToken);
            return true;
        }
    }
}
