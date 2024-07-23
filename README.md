Requires .NET 8 to compile. Once compiled as standalone app, does not require .NET to be installed.

How to run:

```sh
cd csharp
dotnet run ../example .
```

Run in Release configuration (faster):

```sh
cd csharp
dotnet run --configuration Release ../example .
```

Run as AOT compiled code (fastest):

```sh
cd csharp
dotnet publish -r win-x64 -c Release
.\bin\Release\net8.0\win-x64\publish\FSPyramid.exe ../example .
```

For Linux:

Replace `OpenCvSharp4.runtime.win` in `FSPyramid.proj` with the equivalent Linux package from nuget.

AOT install only: change `win-x64` to `linux-x64` or `linux-arm` above
